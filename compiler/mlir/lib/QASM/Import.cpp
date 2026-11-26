// OpenQASM 3 (and 2) into the qnova dialect. A recursive-descent parser for the straight-line circuits toolchains
// exchange: register declarations (qubit[n] q; qreg q[n]; bit / creg), the standard gates of stdgates.inc, U, both
// forms of measurement, barrier, and constant angle expressions (pi / 4, -3 * pi / 8, 1.2e-3). Gate definitions,
// control flow, modifiers (ctrl @) and reset are reported, with the line, as not read yet.
//
// Reading builds SSA as Nova's lowering does (docs/QUANTUM.md): each named qubit q[i] holds the value of its wire
// at this point, and a gate replaces the values of the qubits it acts on with its results. So `cx q[0], q[1];` is
// %a1, %b1 = qnova.cx %a0, %b0, and q[0] and q[1] hold %a1 and %b1 after it.
//
// The circuit's output: a `// nova.probs q[0], q[1]` comment (what the exporter writes) is qnova.probs there; without
// one, the probabilities of the first 12 qubits are printed at the end.
#include "QASM/OpenQASM.h"

#include <cctype>
#include <cmath>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "QNova/QNovaOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"

using namespace mlir;

namespace
{
class Parser
{
    public:
        Parser(llvm::StringRef source, llvm::StringRef file, MLIRContext* context)
            : src(source), file(file.str()), context(context), b(context) {}

        OwningOpRef<ModuleOp> run();

    private:
        llvm::StringRef src;
        std::string file;
        MLIRContext* context;
        OpBuilder b;
        size_t pos = 0;
        unsigned line = 1;
        bool failed = false;
        std::map<std::string, std::vector<Value>> qregs;  // each qubit's current value, by register
        std::vector<Value*> order;                         // every qubit, in declaration order
        std::map<std::string, size_t> cregs;
        std::optional<std::vector<Value*>> probs;          // from a `// nova.probs` comment, once seen

        Location loc() { return FileLineColLoc::get(context, file, line, 1); }
        void fail(const std::string& what)
        {
            if (!failed) emitError(loc()) << what;
            failed = true;
        }

        void skip_space();  // and comments, reading a `// nova.probs` one
        bool at_end()
        {
            skip_space();
            return pos >= src.size();
        }
        char peek()
        {
            skip_space();
            return pos < src.size() ? src[pos] : '\0';
        }
        bool accept(char c)
        {
            if (peek() != c) return false;
            pos++;
            return true;
        }
        void expect(char c)
        {
            if (!accept(c)) fail(std::string("expected '") + c + "'");
        }
        std::string word();     // an identifier (or the empty string)
        double number_literal();
        int64_t integer();
        void skip_statement();  // to the next ';'

        double expr();          // constant expressions: + - * / unary, parentheses, numbers, pi, tau, euler
        double term();
        double factor();
        Value* qubit_operand();
        void statement();
        void gate(const std::string& name);
        Value angle(double v) { return b.create<arith::ConstantOp>(loc(), b.getF64FloatAttr(v)); }
};

void Parser::skip_space()
{
    while (pos < src.size())
    {
        char c = src[pos];
        if (c == '\n')
        {
            line++;
            pos++;
        }
        else if (std::isspace(static_cast<unsigned char>(c)))
            pos++;
        else if (src.substr(pos).starts_with("//"))
        {
            size_t end = src.find('\n', pos);
            llvm::StringRef comment = src.slice(pos + 2, end == llvm::StringRef::npos ? src.size() : end).trim();
            pos = end == llvm::StringRef::npos ? src.size() : end;
            if (comment.consume_front("nova.probs") && !probs)  // the exporter's qnova.probs: read its operands
            {
                std::vector<Value*> list;
                for (llvm::StringRef item : llvm::split(comment, ','))
                {
                    item = item.trim();
                    size_t open = item.find('['), close = item.find(']');
                    auto reg = qregs.find(item.take_front(open).str());
                    unsigned index = 0;
                    if (open == llvm::StringRef::npos || close == llvm::StringRef::npos || reg == qregs.end() ||
                        item.slice(open + 1, close).getAsInteger(10, index) || index >= reg->second.size())
                    {
                        fail("cannot read the qubit '" + item.str() + "' of a nova.probs comment");
                        return;
                    }
                    list.push_back(&reg->second[index]);
                }
                probs = list;
                if (!failed)
                {
                    llvm::SmallVector<Value> values;
                    for (Value* v : list) values.push_back(*v);
                    b.create<qnova::ProbsOp>(loc(), values);
                }
            }
        }
        else if (src.substr(pos).starts_with("/*"))
        {
            size_t end = src.find("*/", pos + 2);
            for (size_t k = pos; k < (end == llvm::StringRef::npos ? src.size() : end); k++)
                if (src[k] == '\n') line++;
            pos = end == llvm::StringRef::npos ? src.size() : end + 2;
        }
        else
            break;
    }
}

std::string Parser::word()
{
    skip_space();
    size_t start = pos;
    if (pos < src.size() && (std::isalpha(static_cast<unsigned char>(src[pos])) || src[pos] == '_'))
        while (pos < src.size() && (std::isalnum(static_cast<unsigned char>(src[pos])) || src[pos] == '_')) pos++;
    else if (src.substr(pos).starts_with("π"))
        pos += std::string("π").size();
    return src.slice(start, pos).str();
}

double Parser::number_literal()
{
    skip_space();
    size_t start = pos;
    while (pos < src.size() && (std::isdigit(static_cast<unsigned char>(src[pos])) || src[pos] == '.' ||
                                src[pos] == 'e' || src[pos] == 'E' ||
                                ((src[pos] == '-' || src[pos] == '+') && pos > start &&
                                 (src[pos - 1] == 'e' || src[pos - 1] == 'E'))))
        pos++;
    double v = 0;
    if (src.slice(start, pos).getAsDouble(v)) fail("expected a number");
    return v;
}

int64_t Parser::integer()
{
    double v = number_literal();
    if (v != std::floor(v) || v < 0) fail("expected a non-negative integer");
    return int64_t(v);
}

void Parser::skip_statement()
{
    while (pos < src.size() && src[pos] != ';')
    {
        if (src[pos] == '\n') line++;
        pos++;
    }
    pos++;
}

double Parser::expr()
{
    double v = term();
    while (true)
    {
        if (accept('+')) v += term();
        else if (accept('-')) v -= term();
        else return v;
    }
}

double Parser::term()
{
    double v = factor();
    while (true)
    {
        if (accept('*')) v *= factor();
        else if (accept('/')) v /= factor();
        else return v;
    }
}

double Parser::factor()
{
    if (accept('-')) return -factor();
    if (accept('+')) return factor();
    if (accept('('))
    {
        double v = expr();
        expect(')');
        return v;
    }
    char c = peek();
    if (std::isdigit(static_cast<unsigned char>(c)) || c == '.') return number_literal();
    std::string w = word();
    if (w == "pi" || w == "π") return M_PI;
    if (w == "tau" || w == "τ") return 2 * M_PI;
    if (w == "euler") return M_E;
    fail("cannot read the angle '" + w + "': a constant expression of numbers and pi is expected");
    return 0;
}

// q[i], or q for a register of one qubit
Value* Parser::qubit_operand()
{
    std::string name = word();
    auto reg = qregs.find(name);
    if (reg == qregs.end())
    {
        fail("'" + name + "' is not a qubit register");
        return nullptr;
    }
    if (!accept('['))
    {
        if (reg->second.size() != 1) fail("a gate on the whole register '" + name + "' is not read yet: write q[i]");
        return failed ? nullptr : &reg->second[0];
    }
    int64_t i = integer();
    expect(']');
    if (failed) return nullptr;
    if (i >= int64_t(reg->second.size()))
    {
        fail("qubit " + std::to_string(i) + " of '" + name + "', which has " + std::to_string(reg->second.size()));
        return nullptr;
    }
    return &reg->second[size_t(i)];
}

void Parser::gate(const std::string& name)
{
    std::vector<double> params;
    if (accept('('))
    {
        if (!accept(')'))
        {
            do params.push_back(expr());
            while (accept(','));
            expect(')');
        }
    }
    std::vector<Value*> qs;
    do qs.push_back(qubit_operand());
    while (!failed && accept(','));
    expect(';');
    if (failed) return;

    auto want = [&](size_t nq, size_t np) {
        if (qs.size() == nq && params.size() == np) return true;
        fail("'" + name + "' takes " + std::to_string(nq) + " qubit(s) and " + std::to_string(np) + " angle(s)");
        return false;
    };
    for (size_t i = 0; i < qs.size(); i++)
        for (size_t j = i + 1; j < qs.size(); j++)
            if (qs[i] == qs[j]) return fail("'" + name + "' is given the same qubit twice");

    Location l = loc();
    Value& a = *qs[0];
    auto one = [&](Value v) { a = v; };
    auto rz = [&](double t) { a = b.create<qnova::RzOp>(l, a, angle(t)); };
    if (name == "h" && want(1, 0)) one(b.create<qnova::HOp>(l, a));
    else if (name == "x" && want(1, 0)) one(b.create<qnova::XOp>(l, a));
    else if (name == "y" && want(1, 0)) one(b.create<qnova::YOp>(l, a));
    else if (name == "z" && want(1, 0)) one(b.create<qnova::ZOp>(l, a));
    else if (name == "s" && want(1, 0)) one(b.create<qnova::SOp>(l, a));
    else if (name == "t" && want(1, 0)) one(b.create<qnova::TOp>(l, a));
    else if (name == "sdg" && want(1, 0)) rz(-M_PI / 2);  // up to a global phase, as below
    else if (name == "tdg" && want(1, 0)) rz(-M_PI / 4);
    else if ((name == "id" || name == "i") && want(1, 0)) {}
    else if (name == "rx" && want(1, 1)) one(b.create<qnova::RxOp>(l, a, angle(params[0])));
    else if (name == "ry" && want(1, 1)) one(b.create<qnova::RyOp>(l, a, angle(params[0])));
    else if ((name == "rz" || name == "p" || name == "phase" || name == "u1") && want(1, 1)) rz(params[0]);
    else if ((name == "U" || name == "u" || name == "u3") && want(1, 3))  // U(t, p, l) = rz(p) ry(t) rz(l)
    {
        rz(params[2]);
        a = b.create<qnova::RyOp>(l, a, angle(params[0]));
        rz(params[1]);
    }
    else if ((name == "cx" || name == "CX" || name == "cnot") && want(2, 0))
    {
        auto g = b.create<qnova::CxOp>(l, a, *qs[1]);
        a = g.getAOut();
        *qs[1] = g.getBOut();
    }
    else if (name == "cz" && want(2, 0))
    {
        auto g = b.create<qnova::CzOp>(l, a, *qs[1]);
        a = g.getAOut();
        *qs[1] = g.getBOut();
    }
    else if (name == "swap" && want(2, 0))
    {
        auto g = b.create<qnova::SwapOp>(l, a, *qs[1]);
        a = g.getAOut();
        *qs[1] = g.getBOut();
    }
    else if ((name == "cp" || name == "cphase" || name == "cu1") && want(2, 1))
    {
        auto g = b.create<qnova::CpOp>(l, a, *qs[1], angle(params[0]));
        a = g.getAOut();
        *qs[1] = g.getBOut();
    }
    else if (!failed)
        fail("the gate '" + name + "' is not read yet (the standard gates are: h x y z s t sdg tdg rx ry rz p U cx "
             "cz swap cp)");
}

void Parser::statement()
{
    if (accept(';')) return;  // an empty statement
    std::string w = word();
    if (w.empty()) return fail(std::string("unexpected '") + peek() + "'");
    if (w == "OPENQASM" || w == "include" || w == "barrier") return skip_statement();
    if (w == "qubit" || w == "bit" || w == "qreg" || w == "creg")
    {
        bool quantum = w == "qubit" || w == "qreg";
        int64_t size = 1;
        std::string name;
        if (w == "qubit" || w == "bit")  // qubit[n] name; or qubit name;
        {
            if (accept('['))
            {
                size = integer();
                expect(']');
            }
            name = word();
        }
        else  // qreg name[n];
        {
            name = word();
            expect('[');
            size = integer();
            expect(']');
        }
        expect(';');
        if (failed) return;
        if (!quantum)
        {
            cregs[name] = size_t(size);
            return;
        }
        if (qregs.count(name)) return fail("the register '" + name + "' is declared twice");
        std::vector<Value>& reg = qregs[name];
        for (int64_t k = 0; k < size; k++) reg.push_back(b.create<qnova::AllocOp>(loc()).getQubit());
        for (Value& v : reg) order.push_back(&v);
        return;
    }
    if (w == "measure")  // measure q[i] -> c[j];  (OpenQASM 2)
    {
        Value* q = qubit_operand();
        if (!failed && accept('-')) expect('>');
        skip_statement();
        if (failed) return;
        *q = b.create<qnova::MeasureOp>(loc(), *q).getOut();
        return;
    }
    if (cregs.count(w))  // c[j] = measure q[i];
    {
        if (accept('['))
        {
            integer();
            expect(']');
        }
        expect('=');
        if (word() != "measure") return fail("only a measurement is assigned to a bit here");
        Value* q = qubit_operand();
        expect(';');
        if (failed) return;
        *q = b.create<qnova::MeasureOp>(loc(), *q).getOut();
        return;
    }
    static const char* const unsupported[] = {"gate", "def", "if", "for", "while", "reset", "ctrl", "negctrl", "inv",
                                              "pow", "input", "output", "let", "const", "defcal", "cal", "box"};
    for (const char* u : unsupported)
        if (w == u) return fail("'" + w + "' is not read yet: Nova imports straight-line circuits of standard gates");
    gate(w);
}

OwningOpRef<ModuleOp> Parser::run()
{
    context->loadDialect<qnova::QNovaDialect, func::FuncDialect, arith::ArithDialect>();
    OwningOpRef<ModuleOp> module = ModuleOp::create(FileLineColLoc::get(context, file, 1, 1));
    b.setInsertionPointToEnd(module->getBody());
    auto f = b.create<func::FuncOp>(loc(), "main", b.getFunctionType({}, {}));
    b.setInsertionPointToStart(f.addEntryBlock());
    while (!failed && !at_end() && !probs) statement();
    if (failed) return nullptr;
    if (!probs && !order.empty())  // no nova.probs comment: the probabilities of the first 12 qubits
    {
        llvm::SmallVector<Value> values;
        for (size_t k = 0; k < order.size() && k < 12; k++) values.push_back(*order[k]);
        b.create<qnova::ProbsOp>(loc(), values);
    }
    if (!at_end()) return fail("statements after the nova.probs comment, the qubits' last use"), nullptr;
    b.create<func::ReturnOp>(loc());
    return module;
}
}  // namespace

OwningOpRef<ModuleOp> qasm::importOpenQASM(llvm::StringRef source, llvm::StringRef file, MLIRContext* context)
{
    return Parser(source, file, context).run();
}
