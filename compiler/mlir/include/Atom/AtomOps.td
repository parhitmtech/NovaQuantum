// The atom dialect's instructions. Each has a start time and a duration in nanoseconds ({start, duration}); a site
// is [x, y] on the array's grid.
#ifndef ATOM_OPS
#define ATOM_OPS

include "Atom/AtomDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// The timing every instruction carries
class Atom_TimedOp<string mnemonic> : Atom_Op<mnemonic>;

// atom.load {site = [0, 0]}: an atom placed at a site, in |0>, before the program starts
def Atom_LoadOp : Atom_Op<"load"> {
    let summary = "an atom at a site, in |0>";
    let arguments = (ins DenseI64ArrayAttr:$site);
    let assemblyFormat = "$site attr-dict";
}

// atom.raman [x, y] theta phi : a local Raman pulse, the rotation r(theta, phi) of one atom
def Atom_RamanOp : Atom_TimedOp<"raman"> {
    let summary = "rotate one atom by theta about cos(phi) X + sin(phi) Y (a Raman pulse)";
    let arguments = (ins DenseI64ArrayAttr:$site, F64Attr:$theta, F64Attr:$phi, I64Attr:$start, I64Attr:$duration);
    let assemblyFormat = "$site `theta` $theta `phi` $phi `at` $start `for` $duration attr-dict";
}

// atom.shift [x, y] theta : a phase shift (rz(theta)) of one atom, by a light shift
def Atom_ShiftOp : Atom_TimedOp<"shift"> {
    let summary = "rotate one atom by theta about Z (a light shift)";
    let arguments = (ins DenseI64ArrayAttr:$site, F64Attr:$theta, I64Attr:$start, I64Attr:$duration);
    let assemblyFormat = "$site `theta` $theta `at` $start `for` $duration attr-dict";
}

// atom.rydberg [x0, y0, x1, y1, ...] : one global Rydberg pulse, a cz on every pair listed (each pair's atoms are
// neighbours, and the pairs far apart: the scheduler made sure)
def Atom_RydbergOp : Atom_TimedOp<"rydberg"> {
    let summary = "a global Rydberg pulse: cz on each pair of neighbouring atoms";
    let arguments = (ins DenseI64ArrayAttr:$pairs, I64Attr:$start, I64Attr:$duration);
    let assemblyFormat = "$pairs `at` $start `for` $duration attr-dict";
    let hasVerifier = 1;
}

// atom.move [x0, y0] to [x1, y1] : a tweezer carries the atom from one site to another
def Atom_MoveOp : Atom_TimedOp<"move"> {
    let summary = "move an atom from one site to another";
    let arguments = (ins DenseI64ArrayAttr:$from, DenseI64ArrayAttr:$to, I64Attr:$start, I64Attr:$duration);
    let assemblyFormat = "$from `to` $to `at` $start `for` $duration attr-dict";
}

// %bit = atom.measure [x, y] : fluorescence imaging of one atom: its bit, the atom left in that state
def Atom_MeasureOp : Atom_TimedOp<"measure"> {
    let summary = "measure the atom at a site";
    let arguments = (ins DenseI64ArrayAttr:$site, I64Attr:$start, I64Attr:$duration);
    let results = (outs I1:$bit);
    let assemblyFormat = "$site `at` $start `for` $duration attr-dict";
}

// atom.readout [x0, y0, x1, y1, ...] : the probability of each basis state of these atoms, printed (the output a
// simulation gives; a machine would sample it from repeated runs)
def Atom_ReadoutOp : Atom_TimedOp<"readout"> {
    let summary = "print the probabilities of the atoms' basis states";
    let arguments = (ins DenseI64ArrayAttr:$sites, I64Attr:$start, I64Attr:$duration);
    let assemblyFormat = "$sites `at` $start `for` $duration attr-dict";
}

#endif  // ATOM_OPS
