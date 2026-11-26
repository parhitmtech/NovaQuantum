// Lowering the nova dialect toward LLVM. A program in nova, arith, scf and func becomes LLVM IR in two steps:
//
//   1. --convert-nova-to-runtime (here): each nova op becomes a call to a function of Nova's C runtime
//      (runtime/nova_runtime.c), so what is left is func, arith and scf, which MLIR itself lowers. @main returns an
//      i32 0, as C's main does, so the program links as a C program's entry point.
//   2. MLIR's own conversions: scf to cf (structured ifs and loops to branches between blocks), then arith, cf, func
//      and ub to the llvm dialect, which mlir-translate writes as LLVM IR. --nova-to-llvm runs both steps.
//
// Step 1 uses MLIR's dialect conversion: a ConversionTarget says which ops may remain (legal) and which must go
// (illegal: the whole nova dialect), and one pattern per op rewrites it. If an illegal op has no pattern that
// applies, the conversion fails and names it, so a program that uses what is not lowered yet stops with an error
// instead of producing a binary that is wrong.
//
// Types. A !nova.tensor<f64> becomes an !llvm.ptr, a pointer to the runtime's NovaTensor: its rank and sizes are
// known only at run time, so the runtime owns its layout. The TypeConverter says so once, and MLIR's structural
// conversions apply it everywhere a tensor can appear besides nova ops: function signatures, calls and returns, and
// the results and block arguments of scf.if and scf.while. Each pattern gets its operands already converted (its
// OpAdaptor), so a tensor operand arrives as the pointer.
#include "Nova/Passes.h"

#include "Nova/NovaOps.h"
#include "QNova/QNovaOps.h"
#include "Atom/AtomOps.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

namespace
{
constexpr unsigned MaxRank = 6;  // the runtime's NOVA_MAX_RANK, the interpreter's TENSOR_MAX_RANK

// The runtime function name, in module m: declared (a func.func with no body, private) the first time it is used.
// The declaration goes at the start of the module, before the function being rewritten.
mlir::func::FuncOp runtime_function(mlir::PatternRewriter& rewriter, mlir::ModuleOp m, llvm::StringRef name,
                                    mlir::FunctionType type)
{
    if (auto f = m.lookupSymbol<mlir::func::FuncOp>(name)) return f;
    mlir::OpBuilder::InsertionGuard guard(rewriter);  // put the rewriter back where it was afterwards
    rewriter.setInsertionPointToStart(m.getBody());
    auto f = rewriter.create<mlir::func::FuncOp>(m.getLoc(), name, type);
    f.setPrivate();  // a declaration: the runtime library defines it
    return f;
}

// A call to the runtime function name with these arguments, declared from their types
mlir::func::CallOp call_runtime(mlir::PatternRewriter& rewriter, mlir::Operation* op, llvm::StringRef name,
                                mlir::TypeRange results, mlir::ValueRange args)
{
    auto type = rewriter.getFunctionType(mlir::TypeRange(args), results);
    runtime_function(rewriter, op->getParentOfType<mlir::ModuleOp>(), name, type);
    return rewriter.create<mlir::func::CallOp>(op->getLoc(), name, results, args);
}

// The Nova line an op came from (nova-lower gives every op its line), for the runtime's error messages
int32_t line_of(mlir::Location loc)
{
    if (auto file = llvm::dyn_cast<mlir::FileLineColLoc>(loc)) return int32_t(file.getLine());
    if (auto fused = llvm::dyn_cast<mlir::FusedLoc>(loc))  // CSE and canonicalize fuse the locations they merge
    {
        for (mlir::Location l : fused.getLocations())
            if (int32_t line = line_of(l)) return line;
    }
    return 0;
}

mlir::Value i32(mlir::PatternRewriter& rewriter, mlir::Location loc, int32_t v)
{
    return rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getI32IntegerAttr(v));
}

// Sizes or subscripts as the runtime takes them: how many, then MaxRank i64 values, the unused ones 0. False if
// there are more than the runtime's rank allows.
bool index_args(mlir::PatternRewriter& rewriter, mlir::Location loc, mlir::ValueRange indices,
                llvm::SmallVectorImpl<mlir::Value>& args)
{
    if (indices.size() > MaxRank) return false;
    args.push_back(i32(rewriter, loc, int32_t(indices.size())));
    mlir::Type i64 = rewriter.getI64Type();
    for (mlir::Value v : indices) args.push_back(rewriter.create<mlir::arith::IndexCastOp>(loc, i64, v));
    mlir::Value zero = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(0));
    for (size_t k = indices.size(); k < MaxRank; k++) args.push_back(zero);
    return true;
}

// The runtime works in doubles; a tensor of another element type is not lowered yet
bool f64_tensor(mlir::Type t)
{
    auto tensor = llvm::dyn_cast<nova::TensorType>(t);
    return tensor && tensor.getElementType().isF64();
}

// nova.print %x : i32 is func.call @nova_print_i32(%x), one runtime function per type, each printing as the
// interpreter does: an int (and a bool, as 1 or 0) with %d, a float or double with %f, a tensor as its rows.
struct PrintLowering : mlir::OpConversionPattern<nova::PrintOp>
{
    using OpConversionPattern::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(nova::PrintOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        mlir::Value x = adaptor.getValue();
        mlir::Type t = x.getType();
        llvm::StringRef name;
        if (t.isInteger(1))  // a bool prints as the int 1 or 0
        {
            t = rewriter.getI32Type();
            x = rewriter.create<mlir::arith::ExtUIOp>(op.getLoc(), t, x);
        }
        if (f64_tensor(op.getValue().getType())) name = "nova_print_tensor";  // x is the converted pointer
        else if (t.isInteger(32)) name = "nova_print_i32";
        else if (t.isF32()) name = "nova_print_f32";
        else if (t.isF64()) name = "nova_print_f64";
        else return rewriter.notifyMatchFailure(op, "no runtime print for this type yet");
        call_runtime(rewriter, op, name, {}, x);
        rewriter.eraseOp(op);
        return mlir::success();
    }
};

// nova.zeros %r, %c is func.call @nova_zeros(line, 2, r, c, 0, 0, 0, 0); nova.ones likewise
template <typename Op>
struct FillLowering : mlir::OpConversionPattern<Op>
{
    FillLowering(const mlir::TypeConverter& converter, mlir::MLIRContext* context, llvm::StringRef name)
        : mlir::OpConversionPattern<Op>(converter, context), name(name) {}
    llvm::StringRef name;

    mlir::LogicalResult matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        if (!f64_tensor(op.getType())) return rewriter.notifyMatchFailure(op, "not a tensor of doubles");
        llvm::SmallVector<mlir::Value> args{i32(rewriter, op.getLoc(), line_of(op.getLoc()))};
        if (!index_args(rewriter, op.getLoc(), adaptor.getSizes(), args))
            return rewriter.notifyMatchFailure(op, "more dimensions than the runtime takes");
        mlir::Type ptr = this->getTypeConverter()->convertType(op.getType());
        rewriter.replaceOp(op, call_runtime(rewriter, op, name, ptr, args).getResults());
        return mlir::success();
    }
};

// Two tensors in, a new one out: nova.add %a, %b is func.call @nova_add(line, a, b); sub, mul, div and matmul too.
// The line is for the runtime's shape errors.
template <typename Op>
struct BinaryLowering : mlir::OpConversionPattern<Op>
{
    BinaryLowering(const mlir::TypeConverter& converter, mlir::MLIRContext* context, llvm::StringRef name)
        : mlir::OpConversionPattern<Op>(converter, context), name(name) {}
    llvm::StringRef name;

    mlir::LogicalResult matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        if (!f64_tensor(op.getType())) return rewriter.notifyMatchFailure(op, "not a tensor of doubles");
        mlir::Value line = i32(rewriter, op.getLoc(), line_of(op.getLoc()));
        mlir::Type ptr = this->getTypeConverter()->convertType(op.getType());
        auto call = call_runtime(rewriter, op, name, ptr, {line, adaptor.getLhs(), adaptor.getRhs()});
        rewriter.replaceOp(op, call.getResults());
        return mlir::success();
    }
};

// A tensor and a number: nova.scale %t, %k is func.call @nova_scale(t, k); nova.add_scalar likewise. Neither can
// fail, so neither takes a line.
template <typename Op>
struct ScalarLowering : mlir::OpConversionPattern<Op>
{
    ScalarLowering(const mlir::TypeConverter& converter, mlir::MLIRContext* context, llvm::StringRef name)
        : mlir::OpConversionPattern<Op>(converter, context), name(name) {}
    llvm::StringRef name;

    mlir::LogicalResult matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        if (!f64_tensor(op.getType())) return rewriter.notifyMatchFailure(op, "not a tensor of doubles");
        mlir::Type ptr = this->getTypeConverter()->convertType(op.getType());
        auto call = call_runtime(rewriter, op, name, ptr, {adaptor.getTensor(), adaptor.getScalar()});
        rewriter.replaceOp(op, call.getResults());
        return mlir::success();
    }
};

// nova.get %t[%i, %j] is func.call @nova_get(line, t, 2, i, j, 0, 0, 0, 0) : f64
struct GetLowering : mlir::OpConversionPattern<nova::GetOp>
{
    using OpConversionPattern::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(nova::GetOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        if (!f64_tensor(op.getTensor().getType())) return rewriter.notifyMatchFailure(op, "not a tensor of doubles");
        llvm::SmallVector<mlir::Value> args{i32(rewriter, op.getLoc(), line_of(op.getLoc())), adaptor.getTensor()};
        if (!index_args(rewriter, op.getLoc(), adaptor.getIndices(), args))
            return rewriter.notifyMatchFailure(op, "more subscripts than the runtime takes");
        rewriter.replaceOp(op, call_runtime(rewriter, op, "nova_get", rewriter.getF64Type(), args).getResults());
        return mlir::success();
    }
};

// nova.set %x, %t[%i, %j] is func.call @nova_set(line, t, x, 2, i, j, 0, 0, 0, 0)
struct SetLowering : mlir::OpConversionPattern<nova::SetOp>
{
    using OpConversionPattern::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(nova::SetOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        if (!f64_tensor(op.getTensor().getType())) return rewriter.notifyMatchFailure(op, "not a tensor of doubles");
        llvm::SmallVector<mlir::Value> args{i32(rewriter, op.getLoc(), line_of(op.getLoc())), adaptor.getTensor(),
                                            adaptor.getValue()};
        if (!index_args(rewriter, op.getLoc(), adaptor.getIndices(), args))
            return rewriter.notifyMatchFailure(op, "more subscripts than the runtime takes");
        call_runtime(rewriter, op, "nova_set", {}, args);
        rewriter.eraseOp(op);
        return mlir::success();
    }
};

// nova.check_divisor %b is func.call @nova_check_divisor(line, b), which stops the program if b is 0
struct CheckDivisorLowering : mlir::OpConversionPattern<nova::CheckDivisorOp>
{
    using OpConversionPattern::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(nova::CheckDivisorOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        if (!adaptor.getDivisor().getType().isInteger(32)) return rewriter.notifyMatchFailure(op, "not an i32");
        mlir::Value line = i32(rewriter, op.getLoc(), line_of(op.getLoc()));
        call_runtime(rewriter, op, "nova_check_divisor", {}, {line, adaptor.getDivisor()});
        rewriter.eraseOp(op);
        return mlir::success();
    }
};

// A tensor variable before anything is assigned holds ub.poison of the tensor type; it becomes poison of the
// pointer type, which no path reads
struct PoisonLowering : mlir::OpConversionPattern<mlir::ub::PoisonOp>
{
    using OpConversionPattern::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(mlir::ub::PoisonOp op, OpAdaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        rewriter.replaceOpWithNewOp<mlir::ub::PoisonOp>(op, getTypeConverter()->convertType(op.getType()));
        return mlir::success();
    }
};

// ─── qnova: circuits through the runtime's statevector simulator (runtime/nova_quantum.c) ───

// A qubit is an i32, its index in the simulator. A gate is a call that updates the simulator's state; the qubits it
// gives are the same indices it was given (qubit operands come first, then any angles), so its results are its
// first operands: qnova.cz %a, %b is func.call @nova_q_cz(a, b), and its results are a and b.
template <typename Op>
struct GateLowering : mlir::OpConversionPattern<Op>
{
    GateLowering(const mlir::TypeConverter& converter, mlir::MLIRContext* context, llvm::StringRef name)
        : mlir::OpConversionPattern<Op>(converter, context), name(name) {}
    llvm::StringRef name;

    mlir::LogicalResult matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        mlir::ValueRange operands = adaptor.getOperands();
        call_runtime(rewriter, op, name, {}, operands);
        rewriter.replaceOp(op, operands.take_front(op->getNumResults()));
        return mlir::success();
    }
};

// %q = qnova.alloc is %q = func.call @nova_q_alloc() : i32, the new qubit's index
struct AllocLowering : mlir::OpConversionPattern<qnova::AllocOp>
{
    using OpConversionPattern::OpConversionPattern;
    mlir::LogicalResult matchAndRewrite(qnova::AllocOp op, OpAdaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        rewriter.replaceOp(op, call_runtime(rewriter, op, "nova_q_alloc", rewriter.getI32Type(), {}).getResults());
        return mlir::success();
    }
};

// A move carries an atom: it changes no state, so in the simulator it is nothing
struct MoveLowering : mlir::OpConversionPattern<qnova::MoveOp>
{
    using OpConversionPattern::OpConversionPattern;
    mlir::LogicalResult matchAndRewrite(qnova::MoveOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        rewriter.replaceOp(op, adaptor.getIn());
        return mlir::success();
    }
};

// %bit, %q1 = qnova.measure %q0: the runtime returns the outcome as an i32, truncated to the i1; the qubit is the same
struct MeasureLowering : mlir::OpConversionPattern<qnova::MeasureOp>
{
    using OpConversionPattern::OpConversionPattern;
    mlir::LogicalResult matchAndRewrite(qnova::MeasureOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        mlir::Value outcome =
            call_runtime(rewriter, op, "nova_q_measure", rewriter.getI32Type(), adaptor.getIn()).getResult(0);
        mlir::Value bit = rewriter.create<mlir::arith::TruncIOp>(op.getLoc(), rewriter.getI1Type(), outcome);
        rewriter.replaceOp(op, {bit, adaptor.getIn()});
        return mlir::success();
    }
};

// qnova.probs %a, %b is func.call @nova_q_probs(2, a, b, 0, ...): how many, then twelve indices
struct ProbsLowering : mlir::OpConversionPattern<qnova::ProbsOp>
{
    using OpConversionPattern::OpConversionPattern;
    mlir::LogicalResult matchAndRewrite(qnova::ProbsOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        mlir::ValueRange qubits = adaptor.getQubits();
        llvm::SmallVector<mlir::Value> args{i32(rewriter, op.getLoc(), int32_t(qubits.size()))};
        args.append(qubits.begin(), qubits.end());
        mlir::Value zero = i32(rewriter, op.getLoc(), 0);
        while (args.size() < 13) args.push_back(zero);
        call_runtime(rewriter, op, "nova_q_probs", {}, args);
        rewriter.eraseOp(op);
        return mlir::success();
    }
};

// ─── atom: a neutral-atom machine's instructions, run by site on the same simulator ───

// The site and angles of an instruction as constants, its arguments in the runtime's order
llvm::SmallVector<mlir::Value> coordinates(mlir::PatternRewriter& rewriter, mlir::Location loc,
                                           llvm::ArrayRef<int64_t> xy)
{
    llvm::SmallVector<mlir::Value> args;
    for (int64_t v : xy) args.push_back(i32(rewriter, loc, int32_t(v)));
    return args;
}

mlir::Value f64(mlir::PatternRewriter& rewriter, mlir::Location loc, double v)
{
    return rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getF64FloatAttr(v));
}

// atom.load [x, y] is func.call @nova_hw_load(x, y); atom.raman, atom.shift and atom.move likewise, by site
struct AtomLowering : mlir::RewritePattern
{
    AtomLowering(mlir::MLIRContext* context) : RewritePattern(MatchAnyOpTypeTag(), 1, context) {}

    mlir::LogicalResult matchAndRewrite(mlir::Operation* op, mlir::PatternRewriter& rewriter) const override
    {
        mlir::Location loc = op->getLoc();
        if (auto load = llvm::dyn_cast<atom::LoadOp>(op))
            call_runtime(rewriter, op, "nova_hw_load", {}, coordinates(rewriter, loc, load.getSite()));
        else if (auto raman = llvm::dyn_cast<atom::RamanOp>(op))
        {
            auto args = coordinates(rewriter, loc, raman.getSite());
            args.push_back(f64(rewriter, loc, raman.getTheta().convertToDouble()));
            args.push_back(f64(rewriter, loc, raman.getPhi().convertToDouble()));
            call_runtime(rewriter, op, "nova_hw_raman", {}, args);
        }
        else if (auto shift = llvm::dyn_cast<atom::ShiftOp>(op))
        {
            auto args = coordinates(rewriter, loc, shift.getSite());
            args.push_back(f64(rewriter, loc, shift.getTheta().convertToDouble()));
            call_runtime(rewriter, op, "nova_hw_shift", {}, args);
        }
        else if (auto rydberg = llvm::dyn_cast<atom::RydbergOp>(op))  // one pulse: a cz on each of its pairs
        {
            llvm::ArrayRef<int64_t> p = rydberg.getPairs();
            for (size_t k = 0; k + 3 < p.size(); k += 4)
                call_runtime(rewriter, op, "nova_hw_rydberg", {}, coordinates(rewriter, loc, p.slice(k, 4)));
        }
        else if (auto move = llvm::dyn_cast<atom::MoveOp>(op))
        {
            auto args = coordinates(rewriter, loc, move.getFrom());
            args.append(coordinates(rewriter, loc, move.getTo()));
            call_runtime(rewriter, op, "nova_hw_move", {}, args);
        }
        else if (auto measure = llvm::dyn_cast<atom::MeasureOp>(op))
        {
            mlir::Value outcome = call_runtime(rewriter, op, "nova_hw_measure", rewriter.getI32Type(),
                                               coordinates(rewriter, loc, measure.getSite()))
                                      .getResult(0);
            rewriter.replaceOp(op, rewriter.create<mlir::arith::TruncIOp>(loc, rewriter.getI1Type(), outcome));
            return mlir::success();
        }
        else if (auto readout = llvm::dyn_cast<atom::ReadoutOp>(op))  // how many sites, then 12 (x, y) pairs
        {
            llvm::ArrayRef<int64_t> s = readout.getSites();
            llvm::SmallVector<mlir::Value> args{i32(rewriter, loc, int32_t(s.size() / 2))};
            args.append(coordinates(rewriter, loc, s));
            while (args.size() < 25) args.push_back(i32(rewriter, loc, 0));
            call_runtime(rewriter, op, "nova_hw_readout", {}, args);
        }
        else
            return mlir::failure();
        rewriter.eraseOp(op);
        return mlir::success();
    }
};

// Nova's @main returns nothing; C's main returns an int, the program's exit status. So @main returns 0, from each
// of its returns. (Not a conversion pattern: it changes a function's type, not an op of the nova dialect.)
void make_entry(mlir::ModuleOp m)
{
    auto main = m.lookupSymbol<mlir::func::FuncOp>("main");
    if (!main || main.getNumResults() != 0 || main.isExternal()) return;
    mlir::OpBuilder b(m.getContext());
    main.setType(b.getFunctionType(main.getArgumentTypes(), {b.getI32Type()}));
    main.walk([&](mlir::func::ReturnOp ret) {
        b.setInsertionPoint(ret);
        mlir::Value zero = b.create<mlir::arith::ConstantOp>(ret.getLoc(), b.getI32IntegerAttr(0));
        b.create<mlir::func::ReturnOp>(ret.getLoc(), zero);
        ret.erase();
    });
}

struct ConvertNovaToRuntime : mlir::PassWrapper<ConvertNovaToRuntime, mlir::OperationPass<mlir::ModuleOp>>
{
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertNovaToRuntime)

    llvm::StringRef getArgument() const override { return "convert-nova-to-runtime"; }
    llvm::StringRef getDescription() const override
    {
        return "Lower nova ops to calls to Nova's C runtime, and make @main return an i32";
    }
    // The dialects the pass creates ops or types of, which the context must load before it runs
    void getDependentDialects(mlir::DialectRegistry& registry) const override
    {
        registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect, mlir::LLVM::LLVMDialect>();
    }

    void runOnOperation() override
    {
        mlir::ModuleOp m = getOperation();
        mlir::MLIRContext* context = &getContext();
        make_entry(m);

        // A tensor is a pointer to the runtime's NovaTensor; every other type stays as it is
        mlir::TypeConverter converter;
        converter.addConversion([](mlir::Type t) { return t; });
        converter.addConversion([](nova::TensorType t) -> mlir::Type {
            return mlir::LLVM::LLVMPointerType::get(t.getContext());
        });
        converter.addConversion([](qnova::QubitType t) -> mlir::Type {  // a qubit is its index in the simulator
            return mlir::IntegerType::get(t.getContext(), 32);
        });

        mlir::ConversionTarget target(*context);
        target.addIllegalDialect<nova::NovaDialect, qnova::QNovaDialect, atom::AtomDialect>();  // all must go
        target.addLegalDialect<mlir::arith::ArithDialect, mlir::func::FuncDialect, mlir::scf::SCFDialect,
                               mlir::ub::UBDialect, mlir::LLVM::LLVMDialect>();
        // ... and no tensor type may remain: an op is legal only once its types are converted
        target.addDynamicallyLegalOp<mlir::func::FuncOp>([&](mlir::func::FuncOp f) {
            return converter.isSignatureLegal(f.getFunctionType()) && converter.isLegal(&f.getBody());
        });
        target.addDynamicallyLegalOp<mlir::func::CallOp, mlir::func::ReturnOp, mlir::ub::PoisonOp>(
            [&](mlir::Operation* op) { return converter.isLegal(op); });

        mlir::RewritePatternSet patterns(context);
        patterns.add<PrintLowering, GetLowering, SetLowering, PoisonLowering, CheckDivisorLowering>(converter,
                                                                                                   context);
        patterns.add<FillLowering<nova::ZerosOp>>(converter, context, "nova_zeros");
        patterns.add<FillLowering<nova::OnesOp>>(converter, context, "nova_ones");
        patterns.add<BinaryLowering<nova::MatmulOp>>(converter, context, "nova_matmul");
        patterns.add<BinaryLowering<nova::AddOp>>(converter, context, "nova_add");
        patterns.add<BinaryLowering<nova::SubOp>>(converter, context, "nova_sub");
        patterns.add<BinaryLowering<nova::MulOp>>(converter, context, "nova_mul");
        patterns.add<BinaryLowering<nova::DivOp>>(converter, context, "nova_div");
        patterns.add<ScalarLowering<nova::ScaleOp>>(converter, context, "nova_scale");
        patterns.add<ScalarLowering<nova::AddScalarOp>>(converter, context, "nova_add_scalar");
        // qnova: each gate a call to the simulator, by name
        patterns.add<AllocLowering, MoveLowering, MeasureLowering, ProbsLowering>(converter, context);
        patterns.add<AtomLowering>(context);  // the hardware instructions, by site
        patterns.add<GateLowering<qnova::HOp>>(converter, context, "nova_q_h");
        patterns.add<GateLowering<qnova::XOp>>(converter, context, "nova_q_x");
        patterns.add<GateLowering<qnova::YOp>>(converter, context, "nova_q_y");
        patterns.add<GateLowering<qnova::ZOp>>(converter, context, "nova_q_z");
        patterns.add<GateLowering<qnova::SOp>>(converter, context, "nova_q_s");
        patterns.add<GateLowering<qnova::TOp>>(converter, context, "nova_q_t");
        patterns.add<GateLowering<qnova::RxOp>>(converter, context, "nova_q_rx");
        patterns.add<GateLowering<qnova::RyOp>>(converter, context, "nova_q_ry");
        patterns.add<GateLowering<qnova::RzOp>>(converter, context, "nova_q_rz");
        patterns.add<GateLowering<qnova::ROp>>(converter, context, "nova_q_r");
        patterns.add<GateLowering<qnova::CxOp>>(converter, context, "nova_q_cx");
        patterns.add<GateLowering<qnova::CzOp>>(converter, context, "nova_q_cz");
        patterns.add<GateLowering<qnova::CpOp>>(converter, context, "nova_q_cp");
        patterns.add<GateLowering<qnova::SwapOp>>(converter, context, "nova_q_swap");
        // Signatures, calls, returns, and scf's results and block arguments, with the converter's types
        mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(patterns, converter);
        mlir::populateCallOpTypeConversionPattern(patterns, converter);
        mlir::populateReturnOpTypeConversionPattern(patterns, converter);
        mlir::scf::populateSCFStructuralTypeConversionsAndLegality(converter, patterns, target);

        // Partial: ops the target does not name may stay; an illegal op left over fails the pass
        if (mlir::failed(mlir::applyPartialConversion(m, target, std::move(patterns)))) signalPassFailure();
    }
};
}  // namespace

std::unique_ptr<mlir::Pass> nova::createConvertNovaToRuntimePass()
{
    return std::make_unique<ConvertNovaToRuntime>();
}

// nova, arith, scf and func, all the way to the llvm dialect: the input mlir-translate --mlir-to-llvmir takes
void nova::buildNovaToLLVMPipeline(mlir::OpPassManager& pm)
{
    pm.addPass(mlir::createCanonicalizerPass());  // folds constants, and drops nova.check_divisor of one other than 0
    pm.addPass(mlir::createCSEPass());
    pm.addPass(nova::createConvertNovaToRuntimePass());
    pm.addPass(mlir::createConvertSCFToCFPass());  // scf.if, scf.while: blocks and branches
    pm.addPass(mlir::createArithToLLVMConversionPass());
    pm.addPass(mlir::createConvertControlFlowToLLVMPass());
    pm.addPass(mlir::createConvertFuncToLLVMPass());
    pm.addPass(mlir::createUBToLLVMConversionPass());
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());  // the casts the conversions leave between them
}

void nova::registerPasses()
{
    mlir::PassRegistration<ConvertNovaToRuntime>();
    mlir::PassPipelineRegistration<>("nova-to-llvm", "Lower a Nova program to the llvm dialect",
                                     buildNovaToLLVMPipeline);
}
