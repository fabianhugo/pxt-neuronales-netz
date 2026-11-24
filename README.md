# Neural network over radio

A small neural network spread across several Calliope mini / micro:bit boards: **one board is
one neuron**. Every board runs this same file — a board learns *which* neuron it is over radio,
so all of them take the identical hex.

This is the radio version of the original cabled build. There are **no wires between the
boards at all**; values travel over radio group 1.

## How a value gets from one neuron to the next

With cables, the **pin** decided which weight a value was multiplied by: a value arriving on P0
was weight 0, P1 was weight 1, and so on.

Radio is a broadcast — every value reaches every board, and no wire says where it belongs. So
each neuron is told the **names** of the neurons feeding it, in weight order:

```
"y0s0:h0"   input slot 0 of y0 comes from h0
"y0s1:h1"   input slot 1 of y0 comes from h1
"y0s2:h2"   input slot 2 of y0 comes from h2
```

A neuron then announces its result to everybody:

```
"h1o" = 0.93        h1's output for this round
```

and each listener looks the sender's name up in its own source list. If `h1` is `y0`'s slot 1,
the value lands in `inputs[1]` and is multiplied by `w1`.

**The source list plays exactly the role the wiring used to**, and it must match the weight
order the training page exported — the same rule as before, now enforced by a table instead of
by cables. Swap two names and the network computes with the weights crossed over: exactly the
failure swapping two cables used to cause, and just as silent.

## Setting it up

1. Flash this onto every board.
2. Read each board's serial number: a board that has no neuron name yet shows it on its display,
   scrolling across, over and over. It is also printed to the USB serial log for a few
   seconds after power-on, and again on button **B**. Enter it in the web page's board table.
3. Plug USB into **any one** board and press **Neuronales Netz senden** on the web page. The page
   sends the network to that board over USB, and that board pushes identity, weights, bias,
   wiring, choices and pictures to everyone over radio.
4. Press **A+B** on an input board, or send `CALC` over serial, to run a round.

The firmware carries **no built-in network and no serial numbers**: every table starts empty and
is filled by the web page. The bridge board keeps them in RAM only, so after a power cut it must
be sent the network again before **A+B** can re-push it. (Every other board keeps its own part in
flash — see below.)

The wiring (`netSources`) is rebuilt automatically by the `ARCH` command: a fully connected
network's sources follow from the layer sizes alone, so the page never has to send it.

## Buttons

| | input (`x`) boards | every other board |
|---|---|---|
| **A** | previous choice | print what this neuron knows |
| **B** | next choice | the same, plus the whole board table |
| **A+B** | start a round | push the network last sent by the web page out over radio |

## Saved settings

A board that has been set up once comes back ready after a power cut — weights, bias **and the
wiring** are kept in flash. With cables the wiring survived a power cut by being physical; over
radio it is data, so it is stored like any other parameter.

A neuron can have up to **10** inputs. Boards are addressed by two-character names (x0–x9,
h0–h9, y0–y9), so no layer can be wider than that anyway.

Saves are versioned. Version 4 (up to 10 weights) replaced version 3 (4 weights): after flashing
this version, every board needs one fresh push. A save from the cabled version has no sources in it, so it is refused
rather than misread, and the board waits for a fresh push.

## Diagnostics

With no cables to trace, the serial dump (button **A**) is how you see what a board is listening
to:

```
in0: 0.73  received: true   from: h0
in1: 0     received: false  from: h1
in2: 0     received: false  from: (no source)
```

A slot showing `(no source)` inside the neuron's fan-in is why a board never fires: it has a
weight but nothing to multiply it by, so it refuses to arm.
