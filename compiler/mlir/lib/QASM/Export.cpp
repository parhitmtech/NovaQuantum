// A qnova circuit as OpenQASM 3. Each qubit value is followed back to the qubit it was allocated as (an alloc gives
// qubit k of the register q, and a gate's results are its operands' qubits, in order), so the SSA chains become
// the named wires OpenQASM writes: %a2, %b1 = qnova.cx %a1, %b0 is `cx q[0], q[1];`.
//
// Gates map one to one onto stdgates.inc, but for the native r(theta, phi), which is U(theta, phi - pi/2, pi/2 - phi).
// What OpenQASM has no statement for goes into comments, which other tools skip and the importer reads back:
// qnova.probs as `// nova.probs q[0], q[1]`, an atom move as `// nova.move q[3] to [1, 0]`.
#include "QASM/OpenQASM.h"

#include <cmath>
#include <string>

#include "QNova/QNovaOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Matchers.h"
#include "llvm/Support/Format.h"

using namespace mlir;

namespace
{
std::string number(double v)
{
    std::string s;
    llvm::raw_string_ostream os(s);
    os << llvm::format("%.17g", v);
    return s;
}
}  // namespace

LogicalResult qasm::exportOpenQASM(ModuleOp m, llvm::raw_ostream& os)
{
    func::FuncOp f;
    m.walk([&](qnova::AllocOp a) {
        if (!f) f = a->getParentOfType<func::FuncOp>();
    });
    if (!f) return m.emitError("no circuit to export: no qnova.alloc");

    llvm::DenseMap<Value, int> qubit;  // each qubit value's index in q
    int qubits = 0, bits = 0;
    std::string body;
    llvm::raw_string_ostream out(body);
    auto q = [&](Value v) { return "q[" + std::to_string(qubit.lookup(v)) + "]"; };

    WalkResult result = f.walk([&](Operation* op) -> WalkResult {
        if (op == f.getOperation() || isa<func::ReturnOp>(op)) return WalkResult::advance();
        if (op->getNumRegions() > 0)
            return op->emitError("control flow in a circuit cannot be exported to OpenQASM yet"),
                   WalkResult::interrupt();
        if (!isa<qnova::QNovaDialect>(op->getDialect())) return WalkResult::advance();  // constants, and the like

        // A constant angle, as the decimal OpenQASM writes
        bool bad = false;
        auto angle = [&](Value v) {
            FloatAttr attr;
            if (matchPattern(v, m_Constant(&attr))) return attr.getValueAsDouble();
            op->emitError("needs a constant angle to be exported (canonicalize folds computed ones)");
            bad = true;
            return 0.0;
        };
        // The results are the operands' qubits, in order
        auto follow = [&] {
            unsigned k = 0;
            for (Value r : op->getResults())
                if (isa<qnova::QubitType>(r.getType())) qubit[r] = qubit.lookup(op->getOperand(k++));
        };

        if (auto a = dyn_cast<qnova::AllocOp>(op))
        {
            qubit[a.getQubit()] = qubits++;
            return WalkResult::advance();
        }
        std::string gate = op->getName().stripDialect().str();
        if (isa<qnova::HOp, qnova::XOp, qnova::YOp, qnova::ZOp, qnova::SOp, qnova::TOp>(op))
            out << gate << " " << q(op->getOperand(0)) << ";\n";
        else if (isa<qnova::RxOp, qnova::RyOp, qnova::RzOp>(op))
            out << gate << "(" << number(angle(op->getOperand(1))) << ") " << q(op->getOperand(0)) << ";\n";
        else if (isa<qnova::CxOp, qnova::CzOp, qnova::SwapOp>(op))
            out << gate << " " << q(op->getOperand(0)) << ", " << q(op->getOperand(1)) << ";\n";
        else if (auto cp = dyn_cast<qnova::CpOp>(op))
            out << "cp(" << number(angle(cp.getLambda())) << ") " << q(cp.getA()) << ", " << q(cp.getB()) << ";\n";
        else if (auto r = dyn_cast<qnova::ROp>(op))  // r(theta, phi) = U(theta, phi - pi/2, pi/2 - phi)
        {
            double theta = angle(r.getTheta()), phi = angle(r.getPhi());
            out << "U(" << number(theta) << ", " << number(phi - M_PI / 2) << ", " << number(M_PI / 2 - phi) << ") "
                << q(r.getIn()) << ";\n";
        }
        else if (auto m = dyn_cast<qnova::MeasureOp>(op))
        {
            if (!m.getBit().use_empty())
                return op->emitError("a measurement whose bit the program uses cannot be exported yet"),
                       WalkResult::interrupt();
            out << "c[" << bits++ << "] = measure " << q(m.getIn()) << ";\n";
        }
        else if (auto mv = dyn_cast<qnova::MoveOp>(op))
            out << "// nova.move " << q(mv.getIn()) << " to [" << mv.getTo()[0] << ", " << mv.getTo()[1] << "]\n";
        else if (auto p = dyn_cast<qnova::ProbsOp>(op))
        {
            out << "// nova.probs";
            for (size_t k = 0; k < p.getQubits().size(); k++) out << (k ? ", " : " ") << q(p.getQubits()[k]);
            out << "\n";
        }
        else
            return op->emitError("has no OpenQASM form"), WalkResult::interrupt();
        if (bad) return WalkResult::interrupt();
        follow();
        return WalkResult::advance();
    });
    if (result.wasInterrupted()) return failure();

    os << "OPENQASM 3.0;\ninclude \"stdgates.inc\";\n";
    os << "qubit[" << qubits << "] q;\n";
    if (bits) os << "bit[" << bits << "] c;\n";
    os << body;
    return success();
}
