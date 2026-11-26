// --qnova-to-atom: a scheduled qnova circuit as the hardware instructions of a neutral-atom machine, with their times.
//
// The last step down Nova's quantum pipeline (docs/QUANTUM.md):
//
//   Nova (qubit, qpu.h(q))  →  qnova logical gates  →  qnova native gates  →  scheduled (sites, moves, layers)
//                                                                                   →  atom: timed instructions
//
// Above this, a qubit is an SSA value; below it, an instruction names a site, as the hardware does. The pass follows
// each qubit value to its atom's site (its alloc's site, then each move's), and writes each layer --qnova-schedule
// made as the instructions that carry it out:
//
//   r(theta, phi)  →  atom.raman: a local Raman pulse, as long as the rotation (raman-pi-ns per pi radians)
//   rz(theta)      →  atom.shift: a light shift, likewise (shift-pi-ns per pi)
//   a layer's cz   →  one atom.rydberg: a single global pulse drives every pair of the layer at once (rydberg-ns)
//   move           →  atom.move: a tweezer move, an overhead to pick up and drop the atom plus the travel
//   measure        →  atom.measure; probs → atom.readout, at the end
//
// The layers run one after another, each as long as its longest instruction, so every instruction gets its start
// and duration in nanoseconds, and the function the total (atom.duration_ns) and how it divides among Raman,
// Rydberg, move and readout time (--atom-timing prints them). The timing model's defaults are illustrative orders of
// magnitude for a neutral-atom machine, not one machine's numbers; each is an option of the pass.
#include "Atom/AtomPasses.h"

#include <cmath>
#include <map>

#include "Atom/AtomOps.h"
#include "QNova/QNovaOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace
{
using Site = std::pair<int64_t, int64_t>;

enum Kind { Raman, Rydberg, Move, Readout, Kinds };
const char* const kindNames[Kinds] = {"raman", "rydberg", "move", "readout"};

struct QNovaToAtom : PassWrapper<QNovaToAtom, OperationPass<func::FuncOp>>
{
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(QNovaToAtom)

    QNovaToAtom() = default;
    QNovaToAtom(const QNovaToAtom& other) : PassWrapper(other) {}

    Option<int64_t> ramanPiNs{*this, "raman-pi-ns", llvm::cl::desc("A Raman rotation by pi, in ns"),
                              llvm::cl::init(1000)};
    Option<int64_t> shiftPiNs{*this, "shift-pi-ns", llvm::cl::desc("A light-shift rotation by pi, in ns"),
                              llvm::cl::init(500)};
    Option<int64_t> rydbergNs{*this, "rydberg-ns", llvm::cl::desc("A global Rydberg pulse, in ns"),
                              llvm::cl::init(400)};
    Option<int64_t> moveSiteNs{*this, "move-site-ns", llvm::cl::desc("Carrying an atom one site's distance, in ns"),
                               llvm::cl::init(10000)};
    Option<int64_t> moveOverheadNs{*this, "move-overhead-ns",
                                   llvm::cl::desc("Picking up and dropping an atom, in ns"), llvm::cl::init(15000)};
    Option<int64_t> measureNs{*this, "measure-ns", llvm::cl::desc("Imaging the atoms, in ns"),
                              llvm::cl::init(1000000)};

    StringRef getArgument() const override { return "qnova-to-atom"; }
    StringRef getDescription() const override
    {
        return "Write a scheduled qnova circuit as a neutral-atom machine's timed instructions (the atom dialect)";
    }
    void getDependentDialects(DialectRegistry& registry) const override { registry.insert<atom::AtomDialect>(); }

    // A rotation's time: proportional to the angle, taken in [0, pi] (a turn the other way is as long)
    int64_t rotation_ns(double theta, int64_t perPi) const
    {
        double a = std::fmod(std::abs(theta), 2 * M_PI);
        a = std::min(a, 2 * M_PI - a);
        return std::max<int64_t>(1, int64_t(std::llround(a / M_PI * double(perPi))));
    }

    void runOnOperation() override
    {
        func::FuncOp f = getOperation();
        bool any = false;
        f.walk([&](qnova::AllocOp) { any = true; });
        if (!any) return;
        if (!f->hasAttr("qnova.depth"))
        {
            f.emitError("the circuit is not scheduled: run --qnova-schedule first (a circuit with control flow is not "
                        "scheduled yet)");
            return signalPassFailure();
        }

        // The circuit's ops in program order, each qubit value's site, and the ops of each layer
        std::vector<Operation*> ops;
        f.walk([&](Operation* op) {
            if (isa<qnova::QNovaDialect>(op->getDialect())) ops.push_back(op);
        });
        llvm::DenseMap<Value, Site> site;
        std::map<int64_t, std::vector<Operation*>> layers;
        std::vector<qnova::AllocOp> allocs;
        std::vector<qnova::ProbsOp> probs;
        for (Operation* op : ops)
        {
            if (auto a = dyn_cast<qnova::AllocOp>(op))
            {
                ArrayRef<int64_t> s = a->getAttrOfType<DenseI64ArrayAttr>("site").asArrayRef();
                site[a.getQubit()] = {s[0], s[1]};
                allocs.push_back(a);
                continue;
            }
            if (auto m = dyn_cast<qnova::MoveOp>(op)) site[m.getOut()] = {m.getTo()[0], m.getTo()[1]};
            else  // a gate's or a measurement's qubits stay where its operands were, in order
            {
                std::vector<Value> in;
                for (Value v : op->getOperands())
                    if (isa<qnova::QubitType>(v.getType())) in.push_back(v);
                size_t k = 0;
                for (Value r : op->getResults())
                    if (isa<qnova::QubitType>(r.getType()) && k < in.size()) site[r] = site.lookup(in[k++]);
            }
            if (auto p = dyn_cast<qnova::ProbsOp>(op))
            {
                probs.push_back(p);
                continue;
            }
            auto layer = op->getAttrOfType<IntegerAttr>("layer");
            if (!layer)
            {
                op->emitError("has no layer: run --qnova-schedule first");
                return signalPassFailure();
            }
            layers[layer.getInt()].push_back(op);
        }

        // A constant angle, which a hardware instruction needs
        bool failed = false;
        auto angle = [&](Operation* op, Value v) {
            FloatAttr attr;
            if (!matchPattern(v, m_Constant(&attr)))
            {
                op->emitError("needs a constant angle on the hardware (canonicalize folds computed ones)");
                failed = true;
                return 0.0;
            }
            return attr.getValueAsDouble();
        };

        // The program: loads first, then each layer's instructions, then the readout, all before the circuit's
        // first op, whose place the program takes
        OpBuilder b(ops.front());
        Location loc = f.getLoc();
        auto sites = [&](std::initializer_list<Site> list) {
            std::vector<int64_t> v;
            for (Site s : list)
            {
                v.push_back(s.first);
                v.push_back(s.second);
            }
            return b.getDenseI64ArrayAttr(v);
        };
        for (qnova::AllocOp a : allocs) b.create<atom::LoadOp>(a.getLoc(), sites({site.lookup(a.getQubit())}));

        int64_t now = 0, instructions = int64_t(allocs.size());
        int64_t byKind[Kinds] = {0, 0, 0, 0};
        for (auto& [index, layerOps] : layers)
        {
            int64_t longest = 0;
            Kind kind = Raman;
            std::vector<int64_t> pairs;
            for (Operation* op : layerOps)
            {
                int64_t ns = 0;
                if (auto r = dyn_cast<qnova::ROp>(op))
                {
                    ns = rotation_ns(angle(op, r.getTheta()), ramanPiNs);
                    b.create<atom::RamanOp>(op->getLoc(), sites({site.lookup(r.getIn())}),
                                            b.getF64FloatAttr(angle(op, r.getTheta())),
                                            b.getF64FloatAttr(angle(op, r.getPhi())), b.getI64IntegerAttr(now),
                                            b.getI64IntegerAttr(ns));
                }
                else if (auto z = dyn_cast<qnova::RzOp>(op))
                {
                    ns = rotation_ns(angle(op, z.getTheta()), shiftPiNs);
                    b.create<atom::ShiftOp>(op->getLoc(), sites({site.lookup(z.getIn())}),
                                            b.getF64FloatAttr(angle(op, z.getTheta())), b.getI64IntegerAttr(now),
                                            b.getI64IntegerAttr(ns));
                }
                else if (auto cz = dyn_cast<qnova::CzOp>(op))
                {
                    kind = Rydberg;
                    Site p = site.lookup(cz.getA()), q = site.lookup(cz.getB());
                    pairs.insert(pairs.end(), {p.first, p.second, q.first, q.second});
                    ns = rydbergNs;
                }
                else if (auto m = dyn_cast<qnova::MoveOp>(op))
                {
                    kind = Move;
                    Site from = site.lookup(m.getIn()), to = {m.getTo()[0], m.getTo()[1]};
                    double distance = std::hypot(double(to.first - from.first), double(to.second - from.second));
                    ns = moveOverheadNs + int64_t(std::llround(distance * double(moveSiteNs)));
                    b.create<atom::MoveOp>(op->getLoc(), sites({from}), sites({to}), b.getI64IntegerAttr(now),
                                           b.getI64IntegerAttr(ns));
                }
                else if (auto m = dyn_cast<qnova::MeasureOp>(op))
                {
                    kind = Readout;
                    ns = measureNs;
                    auto measure = b.create<atom::MeasureOp>(op->getLoc(), b.getI1Type(),
                                                             sites({site.lookup(m.getIn())}), b.getI64IntegerAttr(now),
                                                             b.getI64IntegerAttr(ns));
                    m.getBit().replaceAllUsesWith(measure.getBit());
                }
                else
                {
                    op->emitError("is not a native gate: run --qnova-decompose first");
                    return signalPassFailure();
                }
                if (!isa<qnova::CzOp>(op)) instructions++;
                longest = std::max(longest, ns);
            }
            if (!pairs.empty())  // the layer's cz: one global pulse
            {
                b.create<atom::RydbergOp>(loc, b.getDenseI64ArrayAttr(pairs), b.getI64IntegerAttr(now),
                                          b.getI64IntegerAttr(rydbergNs));
                instructions++;
            }
            byKind[kind] += longest;
            now += longest;
        }
        for (qnova::ProbsOp p : probs)
        {
            std::vector<int64_t> v;
            for (Value q : p.getQubits())
            {
                Site s = site.lookup(q);
                v.push_back(s.first);
                v.push_back(s.second);
            }
            b.create<atom::ReadoutOp>(p.getLoc(), b.getDenseI64ArrayAttr(v), b.getI64IntegerAttr(now),
                                      b.getI64IntegerAttr(measureNs));
            byKind[Readout] += measureNs;
            now += measureNs;
            instructions++;
        }
        if (failed) return signalPassFailure();

        // The circuit is now the program: its qnova ops go, last first (each one's results used only by later ones)
        for (auto it = ops.rbegin(); it != ops.rend(); ++it) (*it)->erase();

        f->setAttr("atom.duration_ns", b.getI64IntegerAttr(now));
        f->setAttr("atom.layers", b.getI64IntegerAttr(int64_t(layers.size())));
        f->setAttr("atom.instructions", b.getI64IntegerAttr(instructions));
        for (int k = 0; k < Kinds; k++)
            f->setAttr(std::string("atom.") + kindNames[k] + "_ns", b.getI64IntegerAttr(byKind[k]));
    }
};

// --atom-timing: one line per program, to stderr:
//   atom-timing @main: 9 layers, 21 instructions, 1050.4 us (raman 6.0 us, rydberg 1.2 us, move 43.2 us, readout ...)
struct Timing : PassWrapper<Timing, OperationPass<func::FuncOp>>
{
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Timing)

    StringRef getArgument() const override { return "atom-timing"; }
    StringRef getDescription() const override { return "Print each hardware program's run time, by instruction kind"; }

    void runOnOperation() override
    {
        func::FuncOp f = getOperation();
        auto total = f->getAttrOfType<IntegerAttr>("atom.duration_ns");
        if (!total) return;
        auto us = [&](const char* name) {
            return double(f->getAttrOfType<IntegerAttr>(name).getInt()) / 1000.0;
        };
        llvm::errs() << "atom-timing @" << f.getSymName() << ": "
                     << f->getAttrOfType<IntegerAttr>("atom.layers").getInt()
                     << " layers, " << f->getAttrOfType<IntegerAttr>("atom.instructions").getInt() << " instructions, "
                     << llvm::format("%.1f", us("atom.duration_ns")) << " us (";
        for (int k = 0; k < Kinds; k++)
            llvm::errs() << (k ? ", " : "") << kindNames[k] << " "
                         << llvm::format("%.1f", us((std::string("atom.") + kindNames[k] + "_ns").c_str())) << " us";
        llvm::errs() << ")\n";
    }
};
}  // namespace

std::unique_ptr<Pass> atom::createQNovaToAtomPass() { return std::make_unique<QNovaToAtom>(); }
std::unique_ptr<Pass> atom::createTimingPass() { return std::make_unique<Timing>(); }

void atom::registerPasses()
{
    registerPass([] { return createQNovaToAtomPass(); });
    registerPass([] { return createTimingPass(); });
}
