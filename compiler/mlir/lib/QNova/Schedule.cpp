// --qnova-schedule: a native circuit placed, routed and scheduled for a neutral-atom machine.
//
// The machine model (docs/QUANTUM.md): atoms sit at the sites of a square grid, one per site. A cz acts on two atoms
// within the interaction radius, here neighbouring sites (Chebyshev distance 1, diagonals included). Unlike a
// superconducting chip, whose qubits are fixed and talk through chains of SWAP gates, a neutral-atom machine can
// carry an atom to another site with an optical tweezer, its state intact. So routing is moving:
//
//   1. Placement. The qubits go, in the order they are allocated, to the black squares of a checkerboard on a grid
//      w sites wide (--width, or one with about twice as many sites as qubits, as a real array has spare sites):
//      each atom then has four diagonal neighbours to interact with and four free sites beside it.
//   2. Routing. Before a cz whose atoms are not neighbours, the second atom moves to the free site next to the
//      first that is nearest to where it is (or the first to the second, if the first is boxed in): a qnova.move
//      {to = [x, y]} on its qubit, before the cz.
//   3. Scheduling. The machine runs the circuit as a sequence of layers, each one kind of operation for many atoms
//      at once: a Raman layer (r, rz), a Rydberg layer (cz), a move layer, a measurement layer. Each op goes to the
//      earliest layer after the ops it depends on (its operands' definitions: SSA def-use is the dependence graph)
//      of its kind. A Rydberg layer drives every atom pair within the blockade radius, so two cz in one layer must
//      be far apart: no atom of one within distance 2 of an atom of the other, or they wait for a later layer.
//
// Each op gets {layer = n}; each function gets qnova.depth (layers), qnova.moves and qnova.move_distance (in sites).
// The model is simple on purpose (one move per cz, no reverse moves, no move parallelism limits), and the pass is
// where a richer machine description would plug in.
#include "QNova/QNovaPasses.h"

#include <cmath>
#include <map>
#include <set>

#include "QNova/QNovaOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace qnova;

namespace
{
using Site = std::pair<int64_t, int64_t>;  // (x, y)

int64_t chebyshev(Site a, Site b) { return std::max(std::abs(a.first - b.first), std::abs(a.second - b.second)); }
double euclid(Site a, Site b) { return std::hypot(double(a.first - b.first), double(a.second - b.second)); }

enum class Kind { Raman, Rydberg, Move, Measure };

struct Layer
{
    Kind kind;
    std::vector<std::pair<Site, Site>> pairs;  // a Rydberg layer's cz, by their atoms' sites
};

struct Schedule : PassWrapper<Schedule, OperationPass<func::FuncOp>>
{
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Schedule)

    Schedule() = default;
    Schedule(const Schedule& other) : PassWrapper(other) {}

    Option<int64_t> width{*this, "width", llvm::cl::desc("Sites per row of the atom grid (0: twice the qubits)"),
                          llvm::cl::init(0)};

    StringRef getArgument() const override { return "qnova-schedule"; }
    StringRef getDescription() const override
    {
        return "Place atoms on a grid, move them next to their cz partners, and schedule ops into hardware layers";
    }

    void runOnOperation() override
    {
        func::FuncOp f = getOperation();
        std::vector<AllocOp> allocs;
        f.walk([&](AllocOp a) { allocs.push_back(a); });
        if (allocs.empty()) return;
        // Only straight-line circuits for now: one with control flow (a gate in an if or a loop, as a program with
        // classical feedback has) is left unscheduled, its gates decomposed and optimized but not placed in layers
        bool controlFlow = false;
        f.walk([&](Operation* op) { controlFlow |= op != f.getOperation() && op->getNumRegions() > 0; });
        if (controlFlow) return;
        // A w x w grid with about twice as many sites as atoms, as a real array has spare sites to move atoms to
        int64_t n = int64_t(allocs.size());
        int64_t w = width > 0 ? int64_t(width) : std::max<int64_t>(2, int64_t(std::ceil(std::sqrt(2.0 * n))));
        OpBuilder b(&getContext());

        // 1. Placement: on a checkerboard, so each atom has four diagonal neighbours to interact with at once and
        // four free sites beside it for a partner to move to. A grid too narrow for that is filled row by row.
        std::vector<Site> sites;
        for (int64_t y = 0; int64_t(sites.size()) < n && y < 4 * n; y++)
            for (int64_t x = 0; x < w && int64_t(sites.size()) < n; x++)
                if ((x + y) % 2 == 0 || w * w < 2 * n) sites.push_back({x, y});
        int64_t h = sites.back().second + 1;
        llvm::DenseMap<Value, Site> site;  // where each qubit value's atom is
        std::set<Site> occupied;
        for (int64_t k = 0; k < n; k++)
        {
            site[allocs[k].getQubit()] = sites[k];
            occupied.insert(sites[k]);
            allocs[k]->setAttr("site", b.getDenseI64ArrayAttr({sites[k].first, sites[k].second}));
        }

        // The free site on the grid next to the atom at 'to' nearest the atom at 'from', if there is one
        auto free_next_to = [&](Site to, Site from) -> std::optional<Site> {
            std::optional<Site> best;
            for (int64_t dx = -1; dx <= 1; dx++)
                for (int64_t dy = -1; dy <= 1; dy++)
                {
                    Site s{to.first + dx, to.second + dy};
                    bool onGrid = s.first >= 0 && s.first < w && s.second >= 0 && s.second < std::max(h, w);
                    if ((dx || dy) && onGrid && !occupied.count(s) && (!best || euclid(s, from) < euclid(*best, from)))
                        best = s;
                }
            return best;
        };

        // 2. Routing, in program order (a topological order of the circuit)
        int64_t moves = 0;
        double distance = 0;
        std::vector<Operation*> ops;
        f.walk([&](Operation* op) { ops.push_back(op); });
        for (Operation* op : ops)
        {
            if (auto cz = dyn_cast<CzOp>(op))
            {
                // Operand 'which' of the cz (0: a, 1: b) carried to site 'to', by a qnova.move just before the cz
                auto move = [&](unsigned which, Site to) {
                    mlir::OpOperand& operand = cz->getOpOperand(which);
                    Site from = site.lookup(operand.get());
                    b.setInsertionPoint(cz);
                    auto m = b.create<MoveOp>(cz.getLoc(), operand.get(),
                                              b.getDenseI64ArrayAttr({to.first, to.second}));
                    operand.set(m.getOut());
                    occupied.erase(from);
                    occupied.insert(to);
                    site[m.getOut()] = to;
                    moves++;
                    distance += euclid(from, to);
                };
                Site sa = site.lookup(cz.getA()), sb = site.lookup(cz.getB());
                if (chebyshev(sa, sb) > 1)
                {
                    // The second atom moves next to the first; if the first is boxed in, the first next to the
                    // second; if both are, the first moves to the free site nearest the second that has room beside
                    // it, and the second joins it there
                    if (std::optional<Site> to = free_next_to(sa, sb)) move(1, *to);
                    else if (std::optional<Site> to = free_next_to(sb, sa)) move(0, *to);
                    else
                    {
                        std::optional<Site> park;
                        for (int64_t x = 0; x < w; x++)
                            for (int64_t y = 0; y < std::max(h, w); y++)
                            {
                                Site s{x, y};
                                if (occupied.count(s) || (park && euclid(s, sb) >= euclid(*park, sb))) continue;
                                occupied.insert(s);  // as if the first atom were there: is there room beside it?
                                bool room = free_next_to(s, sb).has_value();
                                occupied.erase(s);
                                if (room) park = s;
                            }
                        if (!park)
                        {
                            cz.emitError("the atom grid is too full to bring these two atoms together");
                            return signalPassFailure();
                        }
                        move(0, *park);
                        move(1, *free_next_to(*park, sb));
                    }
                }
            }
            // Each qubit an op gives is where its operand was (a move set its own above)
            unsigned qubitIndex = 0;
            std::vector<Value> inputs;
            for (Value v : op->getOperands())
                if (isa<QubitType>(v.getType())) inputs.push_back(v);
            for (Value r : op->getResults())
            {
                if (!isa<QubitType>(r.getType()) || site.count(r)) continue;
                if (qubitIndex < inputs.size()) site[r] = site.lookup(inputs[qubitIndex]);
                qubitIndex++;
            }
        }

        // 3. Scheduling into layers
        std::vector<Layer> layers;
        llvm::DenseMap<Operation*, int64_t> layerOf;
        f.walk([&](Operation* op) {
            std::optional<Kind> kind;
            if (isa<ROp, RzOp, HOp, XOp, YOp, ZOp, SOp, TOp, RxOp, RyOp>(op)) kind = Kind::Raman;
            else if (isa<CzOp, CxOp, CpOp, SwapOp>(op)) kind = Kind::Rydberg;
            else if (isa<MoveOp>(op)) kind = Kind::Move;
            else if (isa<MeasureOp>(op)) kind = Kind::Measure;
            if (!kind) return;
            int64_t earliest = 0;  // after every op that gives this one a qubit
            for (Value v : op->getOperands())
                if (Operation* def = v.getDefiningOp(); def && layerOf.count(def))
                    earliest = std::max(earliest, layerOf[def] + 1);
            std::pair<Site, Site> pair;
            if (*kind == Kind::Rydberg) pair = {site.lookup(op->getOperand(0)), site.lookup(op->getOperand(1))};
            auto fits = [&](const Layer& l) {
                if (l.kind != *kind) return false;
                if (*kind != Kind::Rydberg) return true;
                for (auto& [p, q] : l.pairs)  // no other gate's atoms within the blockade of this one's
                    for (Site s : {p, q})
                        if (chebyshev(s, pair.first) <= 2 || chebyshev(s, pair.second) <= 2) return false;
                return true;
            };
            int64_t at = earliest;
            while (at < int64_t(layers.size()) && !fits(layers[at])) at++;
            if (at == int64_t(layers.size())) layers.push_back(Layer{*kind, {}});
            if (*kind == Kind::Rydberg) layers[at].pairs.push_back(pair);
            layerOf[op] = at;
            op->setAttr("layer", b.getI64IntegerAttr(at));
        });

        f->setAttr("qnova.depth", b.getI64IntegerAttr(int64_t(layers.size())));
        f->setAttr("qnova.moves", b.getI64IntegerAttr(moves));
        f->setAttr("qnova.move_distance", b.getF64FloatAttr(std::round(distance * 1000) / 1000));
    }
};
}  // namespace

std::unique_ptr<Pass> qnova::createSchedulePass() { return std::make_unique<Schedule>(); }
