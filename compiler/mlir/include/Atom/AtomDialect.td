// The atom dialect: timed hardware instructions for a neutral-atom machine (docs/QUANTUM.md)
#ifndef ATOM_DIALECT
#define ATOM_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Atom_Dialect : Dialect {
    let name = "atom";
    let summary = "A neutral-atom machine's instructions: laser pulses and tweezer moves on sites, with their times";
    let description = [{
        The lowest layer of Nova's quantum pipeline. A program is a sequence of instructions on the sites of the atom
        array, each with the time it starts and how long it takes, in nanoseconds: load an atom, a Raman pulse or a
        phase shift on one atom, one global Rydberg pulse for every pair of a layer, a tweezer move, a measurement.
        There are no qubit values here: an instruction names sites, as the hardware does, and the order of the ops
        is the order in time. --qnova-to-atom writes it from a scheduled qnova circuit.
    }];
    let cppNamespace = "::atom";
}

// Every instruction acts on the machine, so none may be removed or reordered: each writes the default resource
class Atom_Op<string mnemonic, list<Trait> traits = []>
    : Op<Atom_Dialect, mnemonic, !listconcat([MemoryEffects<[MemWrite]>], traits)>;

#endif  // ATOM_DIALECT
