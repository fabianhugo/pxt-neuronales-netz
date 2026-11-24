// Not a test suite: in a PXT extension, testFiles are the code that runs when
// the extension is opened as a project in MakeCode, and are NOT compiled into
// a program that merely depends on it. neuron.ts registers all its own handlers
// at load time, so there is nothing to start here.
//
// Open this folder in MakeCode and the board behaves exactly as a flashed
// neuron does: it waits for the sender to tell it who it is.
serial.writeLine("neural-network: neuron firmware loaded")
