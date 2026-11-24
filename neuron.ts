// ============================================================
//  NEURON  --  flash this UNCHANGED onto every neuron board.
//
//  The board learns who it is from the sender: sender.ts holds a table of
//  micro:bit serial number -> neuron name and radios each board its identity,
//  its weights and its bias. Nothing in this file is per-board, so all five
//  boards run the same hex.
//
//  Values between neurons travel over RADIO -- there are no cables at all.
//  Every board is on radio group 1 and hears every other board.
//
//  THE SLOT PROBLEM, and how radio solves it.
//  With cables the PIN decided which weight a value was multiplied by:
//  a value arriving on P0 was weight 0, P1 was weight 1, and so on. Radio is a
//  broadcast -- every value reaches every board and no wire says where it
//  belongs -- so each neuron is instead told the NAMES of the neurons feeding
//  it, in weight order:
//
//      STRING "<name>s<j>:<source>"   source of input slot j
//                                     e.g. "y0s0:h0", "y0s1:h1", "y0s2:h2"
//
//  A neuron then announces its result to everyone as
//
//      "<name>o"    value = this neuron's output    e.g. "h1o" = 0.93
//
//  and each listener looks the sender's name up in its own source list. If h1
//  is y0's slot 1, then "h1o" lands in inputs[1] and is multiplied by w1. The
//  source list therefore plays exactly the role the wiring used to, and it
//  must match the weight order the training page exported -- the same rule as
//  before, now enforced by a table instead of by cables.
//
//  This is why the y boards' own "<name>o" report, which the sender already
//  used to pick a winner, is now the ONE mechanism for all values: a hidden
//  neuron's "h0o" feeds the next layer, and an answer's "y0o" is collected by
//  the sender. Nothing else is needed to move a value between two boards.
//
//  Radio protocol (see sender.ts):
//      STRING "<serial>=<name>"  claims THIS board if the serial matches,
//                                e.g. "-1394225184=h0". A STRING because
//                                radio.sendValue()'s number is a 32-bit float
//                                and cannot hold a 9-10 digit serial.
//      "<name>i"    value = fanIn           e.g. "y0i"  = 3
//      "<name>b"    value = bias            e.g. "h1b"  = 10.15
//      "<name>w<j>" value = weight j        e.g. "h1w0" = -11.38
//      "<name>t"    value = win threshold    e.g. "y0t"  = 0.5
//      STRING "<name>s<j>:<source>"  which neuron feeds input slot j,
//                                e.g. "y0s2:h2". This REPLACES the cable that
//                                used to decide the slot.
//      "<name>o"    value = that neuron's output for this round, broadcast to
//                                everyone; listeners keep it only if the name
//                                is one of their sources.
//      "<name>cr/cg/cb" value = flash colour  e.g. "y1cr" = 40
//      STRING "<name>m<n>:<text>"  chunk n of the message an output
//                                neuron shows when it wins, e.g.
//                                "y1m0:frech". Chunk 0 replaces,
//                                later chunks append.
//      "act"        value = hidden activation  0 sigmoid, 1 tanh, 2 relu,
//                                               3 linear (broadcast to all)
//      "calc"                               start a round
//
//  INPUT (x) boards use the same file. They have no weights; instead the
//  sender gives them a list of selectable values, each with a picture:
//      "<name>n"    value = how many choices        e.g. "x0n" = 4
//      "<name>v<k>" value = choice k's value        e.g. "x0v0" = 0.25
//      "<name>m"    value = mean for scaling        e.g. "x0m" = 0.625
//      "<name>s"    value = spread for scaling      e.g. "x0s" = 0.2795
//                   (0 = send the raw value)
//      STRING "<name>p<k>:<5 packed chars>"  choice k's picture, 5 bits per
//                                        character (code 95 + row value), so
//                                        all 25 pixels fit sendString's
//                                        19-character limit. e.g. "x0p0:zz___".
//  A is previous choice, B is next; the board shows that choice's picture and
//  broadcasts its value as "<name>o" when "calc" arrives, exactly as a hidden
//  neuron broadcasts its activation.
//
//  Buttons:
//      INPUT (x) boards       OTHER boards
//    A   previous choice      print what this neuron knows, to serial
//    B   next choice          the same, plus the whole board table
//   A+B  start a round for    push identity, weights, choices and pictures to
//        the whole network    the whole network (the old sender's job)
//
//  Every board prints its own serial number to the USB log for the first few
//  seconds after power-on, ready to paste into the web page's board table.
// ============================================================

// Without this the board sits on group 0 and never hears the sender.
radio.setGroup(1)

// !! The serial RX buffer defaults to 32 BYTES, and the USB command lines this
// board accepts as bridge run past that ("NEURON 3 -4.7589 -9.2823,9.5337,
// 9.5171" is 38). Overflowing it made serial.readLine() return a CHOPPED line:
// every board serial arrived cut to 9 characters and no board recognised
// itself. Pressing A always worked because that path never touches USB.
// 254 is the most a uint8_t size argument allows.
serial.setRxBufferSize(254)
serial.setTxBufferSize(254)

// ---- identity, assigned over radio -------------------------------------
let ownName = ""          // "h0".."h2", "y0", "y1" -- empty until the sender says
let fanIn = 0             // how many inputs this neuron waits for
let bias = 0
let ownWeights: number[] = [0, 0, 0, 0]
let weightsSeen: boolean[] = [false, false, false, false]
let biasSeen = false
let fanInSeen = false
function hslToHex(h: number, s: number, l: number) {
    h = h % 360
    s = s / 100
    l = l / 100
    c = (1 - Math.abs(2 * l - 1)) * s
    x = c * (1 - Math.abs(h / 60 % 2 - 1))
    m = l - c / 2
    let r, g, b;
    if (h < 60) {
        r = c
        g = x
        b = 0
    } else if (h < 120) {
        r = x
        g = c
        b = 0
    } else if (h < 180) {
        r = 0
        g = c
        b = x
    } else if (h < 240) {
        r = 0
        g = x
        b = c
    } else if (h < 300) {
        r = x
        g = 0
        b = c
    } else {
        r = c
        g = 0
        b = x
    }
    r = Math.round((r + m) * 255)
    g = Math.round((g + m) * 255)
    b = Math.round((b + m) * 255)
    return (r << 16) | (g << 8) | b
}

let m = 0
let x = 0
let c = 0
let l = 0
let s = 0
let h = 0

function rainbow () {
    for (let j = 0; j <= 330; j++) {
        basic.setLedColors(hslToHex(j % 360, 100, 30), hslToHex((j + 30) % 360, 100, 30), hslToHex((j + 60) % 360, 100, 30))
        basic.pause(4)
    }

    basic.turnRgbLedOff()
}
// ---- input (x) boards only ---------------------------------------------
// An x board has no weights. It holds a list of selectable values, each with a
// picture, and sends the selected one when a round starts.
let choiceValues: number[] = []
let choiceValuesSeen: boolean[] = []
// Pictures arrive from the sender as 25-character "0"/"1" strings, one per
// choice. Kept as text and drawn pixel by pixel -- see drawBits().
let choicePics: string[] = []

// How this x board scales its chosen value before putting it on the wire:
// sent = (chosen - myMean) / myStd. myStd of 0 means "send the raw value",
// which is what the sender transmits when the page has scaling switched off.
// Doing it here, rather than folding it into the first layer's weights, keeps
// the maths visible on the boards.
let myMean = 0
let myStd = 0

// ---- output (y) boards only --------------------------------------------
// The message shown when this neuron wins, its flash colour, and the level its
// output must exceed to count as a win. All three come from the sender.
let winMessage = ""
// The picture shown when this neuron wins, as a 25-character "0"/"1" string, or
// "" when none was sent. Same idea as choicePics on an x board: if a picture is
// there it is drawn, otherwise the message is scrolled instead.
let winPicture = ""
let winRed = 0
let winGreen = 0
let winBlue = 0
let outputThreshold = 0.5
// Set by the radio handlers, repainted by the main loop. Drawing must NEVER
// happen inside a handler: it blocks for hundreds of ms, and arriving radio
// packets and softSerial lines are lost outright while it does.
let displayDirty = false
let choiceCount = 0
let choiceIndex = 0

// ---- per-round input state ---------------------------------------------
// received[i] tracks whether inputs[i] has ARRIVED this round. A neuron can
// legitimately send 0.0, so the value itself can never mean "nothing yet".
let inputs: number[] = [0, 0, 0, 0]
let received: boolean[] = [false, false, false, false]
// Which neuron feeds each input slot, in weight order: sources[j] is the name
// whose output is multiplied by ownWeights[j]. This is the radio replacement
// for the cables -- with softSerial the PIN chose the slot, here the SENDER'S
// NAME does. Empty until the sender pushes the wiring, and a slot that is
// still "" can never match an incoming name, so an unconfigured board quietly
// ignores traffic instead of mixing values into the wrong weights.
let sources: string[] = ["", "", "", ""]
let sourcesSeen = 0

// The slot a given neuron's output belongs in, or -1 when that neuron is not
// one of ours. Only the first fanIn slots count: a stale entry left in slot 3
// by an earlier, wider network must not capture a value.
function slotOf(who: string): number {
    for (let j = 0; j <= fanIn - 1 && j <= 3; j++) {
        if (sources[j] == who) {
            return j
        }
    }
    return -1
}
let armed = false         // true between a "calc" request and this neuron firing
let output = 0
// The last value this neuron produced, kept so button B can show it again after
// the screen has been cleared. hasOutput stays false until the first round, so
// an empty bar is never mistaken for a real 0.
let lastOutput = 0
let hasOutput = false
// Stamped a moment ahead whenever a parameter arrives over radio; the main loop
// writes flash once it passes. A push is ~20 messages, and flash has a finite
// erase count, so one write per burst instead of twenty.
let saveWanted = 0

// ============================================================
//  Maths
// ============================================================
// Which activation the hidden layers use, matching the training page's choice:
//   0 sigmoid   1 tanh   2 relu   3 linear
// Sigmoid is the default, as it is on the page.
let actKind = 0

function sigmoid(num: number): number {
    // Clamp first: weights reach +-16, so num can reach about +-40 and
    // Math.exp() would overflow to Infinity on the way to a value that is
    // flat 0 or 1 anyway.
    let z = num
    if (z > 30) {
        z = 30
    }
    if (z < -30) {
        z = -30
    }
    return 1 / (1 + Math.exp(0 - z))
}

// tanh, written from exp because MakeCode has no Math.tanh. The clamp keeps
// exp() from overflowing; beyond +-15 the result is flat -1 or 1 anyway.
function tanhOf(num: number): number {
    let z = num
    if (z > 15) {
        return 1
    }
    if (z < -15) {
        return -1
    }
    let e = Math.exp(2 * z)
    return (e - 1) / (e + 1)
}

// Apply whichever activation this network was trained with.
function activate(z: number): number {
    if (actKind == 1) {
        return tanhOf(z)
    }
    if (actKind == 2) {
        return z > 0 ? z : 0          // relu
    }
    if (actKind == 3) {
        return z                       // linear
    }
    return sigmoid(z)
}

// The name, for the serial logs.
function actName(): string {
    if (actKind == 1) {
        return "tanh"
    }
    if (actKind == 2) {
        return "relu"
    }
    if (actKind == 3) {
        return "linear"
    }
    return "sigmoid"
}

function getNeuronOutput(): number {
    rainbow()
    let result = 0
    for (let i = 0; i <= fanIn - 1; i++) {
        result = result + inputs[i] * ownWeights[i]
    }
    result = result + bias
    // The training page applies the chosen activation to HIDDEN layers only --
    // its output layer is always softmax. The answer boards cannot compute a
    // softmax (it needs every answer at once, and each board sees only its
    // own), but the winner is whichever value is largest and softmax is
    // monotonic, so the comparison is unchanged. Sigmoid is kept on the answer
    // boards because it maps to 0..1, which is what the win threshold and the
    // output bar graph expect.
    if (ownName.charAt(0) == "y") {
        return sigmoid(result)
    }
    return activate(result)
}

function isInputBoard(): boolean {
    return ownName.charAt(0) == "x"
}

// CAUTION: basic.showString / showNumber SCROLL and block long enough to lose
// an arriving softSerial line. Only showLeds / showIcon / clearScreen, which
// are instant, may be used in the hot paths.

// Draw a 25-character "0"/"1" bitmap, row by row. led.plot/unplot are instant;
// basic.showImage would block long enough to lose an arriving softSerial line.
function drawBits(bits: string) {
    for (let y = 0; y <= 4; y++) {
        for (let x = 0; x <= 4; x++) {
            if (bits.charAt(y * 5 + x) == "1") {
                led.plot(x, y)
            } else {
                led.unplot(x, y)
            }
        }
    }
}

// Put the screen back to whatever this board normally shows.
function showState() {
    if (isInputBoard()) {
        showChoice()
    } else if (ownName == "") {
        basic.showString("?")
    } else {
        basic.showString(ownName)
    }
}

function hasChoice(k: number): boolean {
    return k >= 0 && k < choiceValuesSeen.length && choiceValuesSeen[k]
}

// Show the board's state on the 5x5 matrix.
//
// Pictures come from the sender over radio (see sender.ts inputPictures), so a
// board stores no artwork of its own and the set can be changed without
// reflashing. drawBits is instant; nothing here scrolls.
function showChoice() {
    if (ownName == "") {
        basic.showString("?")              // no identity yet
        return
    }
    if (!hasChoice(choiceIndex)) {
        basic.showIcon(IconNames.No, 0)    // this choice never arrived
        return
    }
    if (choiceIndex < choicePics.length && choicePics[choiceIndex] != null) {
        drawBits(choicePics[choiceIndex])
        return
    }
    // Value known but its picture has not arrived yet: show the index as a
    // row of dots so the board is still usable.
    basic.clearScreen()
    for (let i = 0; i <= choiceIndex && i <= 4; i++) {
        led.plot(i, 2)
    }
}



// Report the selection to serial, saying plainly when a value is missing
// instead of printing "undefined".
function reportChoice() {
    if (hasChoice(choiceIndex)) {
        serial.writeValue("sel_" + ownName, choiceValues[choiceIndex])
    } else {
        serial.writeLine("sel_" + ownName + ": MISSING (choice " + choiceIndex +
            " never arrived -- press A on the sender again)")
    }
}


// ============================================================
//  Readiness
// ============================================================
function isConfigured(): boolean {
    if (ownName == "") {
        return false
    }
    // An x board needs choices, not weights.
    if (isInputBoard()) {
        if (choiceCount <= 0) {
            return false
        }
        // Radio is unacknowledged: a dropped "<name>v<k>" leaves a hole, and
        // stepping onto it would send undefined into the network.
        for (let k = 0; k <= choiceCount - 1; k++) {
            if (k >= choiceValuesSeen.length || !choiceValuesSeen[k]) {
                return false
            }
        }
        return true
    }
    if (!fanInSeen || !biasSeen) {
        return false
    }
    for (let i = 0; i <= fanIn - 1; i++) {
        if (!weightsSeen[i]) {
            return false
        }
        // A weight with no source is a connection with no cable: values for
        // that slot would never arrive and the board would wait out the round
        // forever. Radio is unacknowledged, so a dropped "<name>s<j>" is a real
        // possibility -- refusing to arm makes it visible as "?" on the display
        // instead of a network that silently never answers.
        if (sources[i] == "") {
            return false
        }
    }
    return true
}

function hasAllInputs(): boolean {
    for (let i = 0; i <= fanIn - 1; i++) {
        if (!received[i]) {
            return false
        }
    }
    return true
}

function clearRound() {
    for (let i = 0; i <= 3; i++) {
        inputs[i] = 0
        received[i] = false
    }
    armed = false
}

// ============================================================
//  Radio: identity, parameters and the calculate request
// ============================================================
radio.onReceivedValue(function (name, value) {
    // A board acting as sender also listens for the answer boards reporting in.
    collectAnswer(name, value)

    // ---- another neuron's output: "<source>o" ----
    // This is the radio replacement for a value arriving on an input pin. It
    // is checked FIRST and returns immediately, so a neuron output can never
    // fall through into the parameter parsing below -- "h0o" would otherwise
    // be read as a field "o" addressed to a neuron called "h0".
    if (name.length >= 3 && name.charAt(name.length - 1) == "o") {
        let who = name.substr(0, name.length - 1)
        if (acceptInput(who, value)) {
            return
        }
        // Not one of our sources. Still a neuron output, so nothing else can
        // want it -- except the sender's collectAnswer(), which already ran.
        return
    }

    // Broadcasts meant for everyone.
    if (name == "act") {
        // Which activation the hidden layers use. Broadcast, not addressed to a
        // name: every board in the network was trained with the same one.
        actKind = value
        saveWanted = input.runningTime() + 2000
        serial.writeLine("activation: " + actName())
        return
    }
    if (name == "calc") {
        // Do NOT clear inputs here. An x board may already have sent its value
        // before this board processed its own "calc" -- clearing would wipe a
        // value that had already arrived, and the board would then wait forever
        // for something already delivered. Inputs are cleared on COMPLETION
        // (clearRound after firing), so a round starts clean regardless.
        armed = true
        return
    }
    // Everything else is "<neuronName><field>", e.g. "h1w0", "y0b", "y0i".
    // Names are 2 chars, so the field starts at index 2.
    if (name.length < 3) {
        return
    }
    let target = name.substr(0, 2)
    let field = name.substr(2, name.length - 2)

    // Ignore anything addressed to another neuron, and everything at all
    // until the sender has told us who we are.
    if (ownName == "" || target != ownName) {
        return
    }

    if (field == "i") {
        fanIn = value
        fanInSeen = true
        saveWanted = input.runningTime() + 2000
        return
    }
    if (field == "m") {
        myMean = value
        saveWanted = input.runningTime() + 2000
        return
    }
    if (field == "s") {
        myStd = value
        saveWanted = input.runningTime() + 2000
        return
    }
    if (field == "n") {
        // x board: how many selectable choices. Starting a new list clears the
        // old one, so a re-push cannot leave stale values behind.
        choiceCount = value
        choiceIndex = 0
        choiceValues = []
        choiceValuesSeen = []
        choicePics = []
        saveWanted = input.runningTime() + 2000
        return
    }
    if (field == "b") {
        bias = value
        biasSeen = true
        saveWanted = input.runningTime() + 2000
        return
    }
    if (field == "t") {
        // y board: output must exceed this to count as a win
        outputThreshold = value
        saveWanted = input.runningTime() + 2000
        return
    }
    if (field == "cr") {
        winRed = value
        saveWanted = input.runningTime() + 2000
        return
    }
    if (field == "cg") {
        winGreen = value
        saveWanted = input.runningTime() + 2000
        return
    }
    if (field == "cb") {
        winBlue = value
        saveWanted = input.runningTime() + 2000
        return
    }
    if (field.charAt(0) == "v") {
        // x board: choice k's value
        let k = parseInt(field.substr(1, field.length - 1))
        while (choiceValues.length <= k) {
            choiceValues.push(0)
            choiceValuesSeen.push(false)
        }
        choiceValues[k] = value
        choiceValuesSeen[k] = true
        saveWanted = input.runningTime() + 2000
        if (k == choiceIndex) {
            displayDirty = true
        }
        return
    }
    if (field.charAt(0) == "w") {
        let j = parseInt(field.substr(1, field.length - 1))
        if (j >= 0 && j <= 3) {
            ownWeights[j] = value
            weightsSeen[j] = true
            saveWanted = input.runningTime() + 2000
        }
    }
})

// ONE handler for every radio string -- there must only ever be one:
// registering radio.onReceivedString twice REPLACES the first, which silently
// killed identity assignment once already.
//
// Two shapes arrive here, both as TEXT because neither survives
// radio.sendValue()'s 32-bit float (exact only to 2^24):
//   "<serial>=<name>"         identity,  e.g. "-2022602061=x0"
//   "<name>p<k>:<25 chars>"   picture for choice k of an x board
radio.onReceivedString(function (receivedString) {
    // ---- picture: "<name>p<k>:<5 packed chars>" ----
    // The 25 pixels arrive PACKED, one character per row (5 bits each, code
    // 95 + value), because radio.sendString only carries 19 characters and the
    // unpacked form needed 30. See packPicture() in sender.ts.
    // Both the picture and the win message are "<head>:<payload>", so match on
    // the head rather than returning early -- an earlier version returned on
    // ANY colon message whose head was not a picture, which silently swallowed
    // the message chunks.
    let colon = receivedString.indexOf(":")
    let head = colon >= 0 ? receivedString.substr(0, colon) : ""
    let payload = colon >= 0
        ? receivedString.substr(colon + 1, receivedString.length - colon - 1)
        : ""
    let mine = head.length >= 3 && head.substr(0, 2) == ownName

    if (mine && head.charAt(2) == "p" && payload.length >= 5) {
        // "<name>p<k>:<5 packed chars>" -- one character per row, 5 bits each
        // at code 95 + value.
        let k = parseInt(head.substr(3, head.length - 3))
        let bits = ""
        for (let y = 0; y <= 4; y++) {
            let v = payload.charCodeAt(y) - 95
            for (let x = 0; x <= 4; x++) {
                if (v & (1 << x)) {
                    bits = bits + "1"
                } else {
                    bits = bits + "0"
                }
            }
        }
        while (choicePics.length <= k) {
            choicePics.push(null)
        }
        choicePics[k] = bits
        saveWanted = input.runningTime() + 2000
        serial.writeLine("pic " + ownName + " " + k + " <- [" + payload + "]")
        if (k == choiceIndex) {
            displayDirty = true
        }
        return
    }

    // ---- win picture: "<name>q0:<5 packed chars>" ----
    // Same packing as a choice picture: one character per row, 5 bits each at
    // code 95 + value, which keeps it inside the 19-character radio limit.
    if (mine && head.charAt(2) == "q") {
        let bits = ""
        for (let y = 0; y <= 4; y++) {
            let v = payload.charCodeAt(y) - 95
            for (let x = 0; x <= 4; x++) {
                if (v & (1 << x)) {
                    bits = bits + "1"
                } else {
                    bits = bits + "0"
                }
            }
        }
        winPicture = bits
        saveWanted = input.runningTime() + 2000
        serial.writeLine("winpic " + ownName + " <- [" + payload + "]")
        return
    }

    // ---- input source: "<name>s<j>:<source>" ----
    // The radio replacement for the cables: it names the neuron whose output
    // belongs in input slot j, which used to be decided by the pin a wire was
    // plugged into. Sent as a STRING because the payload is a name, not a
    // number, and kept per-slot so a re-push can correct one connection
    // without resending the whole network.
    if (mine && head.charAt(2) == "s") {
        let j = parseInt(head.substr(3, head.length - 3))
        if (j >= 0 && j <= 3) {
            sources[j] = payload
            sourcesSeen = sourcesSeen | (1 << j)
            saveWanted = input.runningTime() + 2000
            serial.writeLine("src " + ownName + " slot " + j + " <- " + payload)
        }
        return
    }

    // ---- win message chunk: "<name>m<n>:<text>" ----
    // Chunked because radio.sendString carries only 19 characters. Chunk 0
    // starts a fresh message, later chunks append, so any length works.
    if (mine && head.charAt(2) == "m") {
        let n = parseInt(head.substr(3, head.length - 3))
        if (n == 0) {
            winMessage = payload
        } else {
            winMessage = winMessage + payload
        }
        saveWanted = input.runningTime() + 2000
        return
    }

    // Any other colon message is not for us.
    if (colon >= 0) {
        return
    }

    // ---- identity: "<serial>=<name>" ----
    let eq = receivedString.indexOf("=")
    if (eq < 0) {
        return
    }
    let wantSerial = receivedString.substr(0, eq)
    let wantName = receivedString.substr(eq + 1, receivedString.length - eq - 1)
    let mySerial = convertToText(control.deviceSerialNumber())
    serial.writeLine("id? want [" + wantSerial + "](" + wantSerial.length +
        ") mine [" + mySerial + "](" + mySerial.length + ") -> " +
        (wantSerial == mySerial ? "MATCH " + wantName : "no"))
    if (wantSerial == mySerial) {
        ownName = wantName
        saveWanted = input.runningTime() + 2000
        displayDirty = true
    } else if (ownName == wantName) {
        // This name now belongs to a different board -- let it go.
        ownName = ""
        displayDirty = true
    }
})

// ============================================================
//  Receiving a value from another neuron
//
//  There is nothing to register here: a neuron's output arrives as the radio
//  value "<source>o" and is picked up by the one radio.onReceivedValue handler
//  above, which calls acceptInput(). That handler must stay the ONLY one --
//  registering onReceivedValue twice replaces the first.
//
//  What used to be three pin handlers is now one name lookup. The old
//  basic.pause(10) those handlers needed is gone with them: it existed to let
//  the bit-banged softSerial driver settle, and the radio driver has its own
//  buffer.
// ============================================================
// Store an incoming neuron output in whichever slot that neuron feeds.
// Returns true when the value was ours, so the caller can stop looking.
function acceptInput(who: string, value: number): boolean {
    let j = slotOf(who)
    if (j < 0) {
        return false
    }
    // A board hears EVERY round from EVERY neuron, including neurons that have
    // already fired while this one was still waiting. Accepting a value only
    // while armed would drop a source that was quicker than this board's own
    // "calc" -- the same race the cabled version hit, and the reason "calc"
    // does not clear the inputs. So take it whenever it arrives; clearRound()
    // after firing is what separates one round from the next.
    inputs[j] = value
    received[j] = true
    serial.writeLine("RX " + who + " -> slot " + j + ": " + value)
    return true
}

// ============================================================
//  Persistence -- survive a power cycle
//
//  settings.writeNumberArray stores each non-integer as a msgpack 0xCB tag,
//  which is a full Float64BE: no precision is lost, and the weights come back
//  bit-identical (checked against every weight in this network).
//
//  If you saw numbers arriving with two decimals, that is the NUMBER->STRING
//  conversion in serial.writeValue / convertToText, not the storage -- the
//  stored value is intact even when the printed one looks rounded.
//
//  Saved after every push so a board that has been set up once comes back ready
//  after a power cut, with no computer and no sender.
// ============================================================
const SAVE_KEY = "nn"
// Bumped whenever the layout of the saved number array changes. A save written
// by an older version is discarded rather than misread.
//
// 3: the radio version. A saved board now also needs its SOURCES -- with cables
// the wiring survived a power cut by being physical, over radio it is data and
// has to be stored like any other parameter. A version-2 save has none, and a
// board restored from one would have weights it could never match a value to,
// so such a save is refused outright and the board waits for a fresh push.
const SAVE_VERSION = 3

// One flat array: name-as-digits are impossible, so the name is stored
// separately as a string and everything numeric goes in one array.
//   [fanIn, bias, w0..w3, myMean, myStd, outputThreshold,
//    winRed, winGreen, winBlue, choiceCount, choiceValues...]
function saveToFlash() {
    if (ownName == "") {
        return
    }
    settings.writeString(SAVE_KEY + "n", ownName)
    settings.writeString(SAVE_KEY + "m", winMessage)
    // Its own key, not a slot in the numeric array: adding a field there is what
    // made an older save unreadable last time, and a string cannot go in it anyway.
    settings.writeString(SAVE_KEY + "q", winPicture)
    // The wiring, as one comma-separated line ("h0,h1,h2"). A string, so it
    // cannot go in the numeric array; one key rather than one per slot, because
    // the slots are only meaningful together.
    let srcLine = ""
    for (let j = 0; j <= 3; j++) {
        srcLine = srcLine + (j > 0 ? "," : "") + sources[j]
    }
    settings.writeString(SAVE_KEY + "s", srcLine)
    // A version marker in slot 0. Without it an older save is indistinguishable
    // from a newer one and its choice values get read as the wrong fields --
    // the first choice was being loaded as the activation.
    let nums = [SAVE_VERSION, fanIn, bias]
    for (let i = 0; i <= 3; i++) {
        nums.push(ownWeights[i])
    }
    nums.push(myMean)
    nums.push(myStd)
    nums.push(outputThreshold)
    nums.push(winRed)
    nums.push(winGreen)
    nums.push(winBlue)
    nums.push(choiceCount)
    nums.push(actKind)
    for (let k = 0; k <= choiceValues.length - 1; k++) {
        nums.push(choiceValues[k])
    }
    settings.writeNumberArray(SAVE_KEY + "v", nums)
    // Pictures are text, so they go in their own keys.
    for (let k = 0; k <= choicePics.length - 1; k++) {
        if (choicePics[k] != null) {
            settings.writeString(SAVE_KEY + "p" + k, choicePics[k])
        }
    }
    serial.writeLine("saved " + ownName + " to flash")
}

function loadFromFlash() {
    let nm = settings.readString(SAVE_KEY + "n")
    if (nm == null || nm == "") {
        return false
    }
    let nums = settings.readNumberArray(SAVE_KEY + "v")
    if (nums == null || nums.length < 14) {
        return false
    }
    // Refuse anything this version does not understand, rather than reading a
    // choice value as an activation and shifting every field after it.
    if (nums[0] != SAVE_VERSION) {
        serial.writeLine("saved data is version " + nums[0] + ", expected " +
            SAVE_VERSION + " -- ignoring it")
        return false
    }
    ownName = nm
    // The wiring. Slot j keeps "" when the line is short or the field empty,
    // which isConfigured() reads as "not wired yet" -- the same state as a
    // board that has never been pushed to, and the safe one.
    let srcLine = settings.readString(SAVE_KEY + "s")
    sources = ["", "", "", ""]
    sourcesSeen = 0
    if (srcLine != null) {
        let cur = ""
        let slot = 0
        for (let i = 0; i <= srcLine.length; i++) {
            // One past the end on purpose: it flushes the last field without
            // repeating the push below.
            let ch = i < srcLine.length ? srcLine.charAt(i) : ","
            if (ch == ",") {
                if (slot <= 3) {
                    sources[slot] = cur
                    if (cur != "") {
                        sourcesSeen = sourcesSeen | (1 << slot)
                    }
                }
                slot = slot + 1
                cur = ""
            } else {
                cur = cur + ch
            }
        }
    }
    fanIn = nums[1]
    fanInSeen = fanIn > 0
    bias = nums[2]
    biasSeen = true
    for (let i = 0; i <= 3; i++) {
        ownWeights[i] = nums[3 + i]
        weightsSeen[i] = i < fanIn
    }
    myMean = nums[7]
    myStd = nums[8]
    outputThreshold = nums[9]
    winRed = nums[10]
    winGreen = nums[11]
    winBlue = nums[12]
    choiceCount = nums[13]
    actKind = nums[14]
    choiceValues = []
    choiceValuesSeen = []
    for (let k = 0; k <= choiceCount - 1; k++) {
        if (15 + k <= nums.length - 1) {
            choiceValues.push(nums[15 + k])
            choiceValuesSeen.push(true)
        }
    }
    let msg = settings.readString(SAVE_KEY + "m")
    winMessage = msg == null ? "" : msg
    let wpic = settings.readString(SAVE_KEY + "q")
    winPicture = wpic == null ? "" : wpic
    choicePics = []
    for (let k = 0; k <= choiceCount - 1; k++) {
        let pic = settings.readString(SAVE_KEY + "p" + k)
        choicePics.push(pic == null ? null : pic)
    }
    serial.writeLine("restored " + ownName + " from flash")
    return true
}

// ============================================================
//  Main loop: fire once per round, when every input has arrived
// ============================================================
basic.forever(function () {
    basic.pause(20)
    // Repaint here rather than in the radio handlers that set displayDirty.
    // basic.showString scrolls for over a second, and a board stuck in it MISSES
    // an arriving softSerial line outright -- bit-banged serial has no buffer.
    // Doing it here keeps handlers instant, and a repaint only ever happens on a
    // state change, never mid-round.
    if (displayDirty) {
        displayDirty = false
        showState()
    }
    // Debounced flash write: once the radio burst has been quiet for 2 s.
    if (saveWanted > 0 && input.runningTime() > saveWanted) {
        saveWanted = 0
        saveToFlash()
    }
    if (!armed || !isConfigured()) {
        return
    }
    // An x board produces a value instead of waiting for one: on "calc" it
    // just sends whichever choice is currently selected.
    if (isInputBoard()) {
        if (!hasChoice(choiceIndex)) {
            serial.writeLine(ownName + ": cannot send, choice " + choiceIndex +
                " is missing")
            armed = false
            return
        }
        // Receivers no longer clear on "calc", so this no longer has to win a
        // race -- a short settle is enough to let every board arm itself.
        basic.pause(50)
        // Scale before sending, when the sender has given this board a mean and
        // spread. The network was trained on scaled values, so the scaling has
        // to happen somewhere; doing it here keeps it visible on the board
        // instead of hidden inside the first layer's weights.
        output = choiceValues[choiceIndex]
        if (myStd != 0) {
            output = (output - myMean) / myStd
            serial.writeLine("chose " + choiceValues[choiceIndex] + " -> scaled " + output)
        }
        serial.writeLine("TX " + ownName + "o: " + output)
        led.plotBarGraph(output, 1)
        basic.pause(1000)
        basic.showIcon(IconNames.ArrowEast)
        // Broadcast, not a wire: every board hears it and the ones that list
        // this x board as a source keep it. Sent BEFORE the display pause so
        // the next layer can start as soon as possible.
        radio.sendValue(ownName + "o", output)
        basic.pause(1000)
        showChoice()
        armed = false
        return
    }
    if (!hasAllInputs()) {
        return
    }
    output = getNeuronOutput()
    lastOutput = output
    hasOutput = true
    serial.writeValue("out_" + ownName, output)
    // A bar, not a scrolling number: instant instead of ~a second of blocking,
    // and the height reads at a glance. The 1 is the top of the scale -- with 0
    // MakeCode auto-scales to the largest value it has seen, which would make
    // the bar mean something different from one round to the next.
    led.plotBarGraph(output, 1)
    // One broadcast serves both purposes: the next layer picks it up as an
    // input (whoever lists this neuron as a source) and, for a y board, the
    // sender picks it up as an answer. This is sent for EVERY neuron now, not
    // just the y boards -- it is how a value crosses to the next layer.
    //
    // Sent BEFORE the display pause: a hidden neuron used to put its value on
    // the wire and then block for 500 ms, which cost the whole next layer half
    // a second per layer. Broadcasting first lets the next layer compute while
    // this board is still showing its bar.
    radio.sendValue(ownName + "o", output)
    basic.pause(500)
    basic.clearScreen()
    // Output neurons announce a win locally.
    if (ownName.charAt(0) == "y") {
        // Announce a win locally, the way nn.ts did: flash this neuron's own
        // colour, then scroll its message. Colour, message and threshold all
        // come from the sender, so they change without reflashing.
        // Blocking is safe here: the round is over and no input is expected
        // until the next "calc".
        if (output > outputThreshold) {
            for (let index = 0; index < 5; index++) {
                basic.setLedColor(basic.rgb(winRed, winGreen, winBlue))
                basic.pause(100)
                basic.turnRgbLedOff()
                basic.pause(100)
            }
            // A picture if one was sent, the words if not. drawBits is instant;
            // showString scrolls, but blocking is safe here -- the round is over.
            if (winPicture != "") {
                drawBits(winPicture)
                basic.pause(1500)
            } else if (winMessage != "") {
                basic.showString(winMessage)
            }
            displayDirty = true      // go back to showing the board's name
        }
    }
    clearRound()
})


// ============================================================
//  SENDER ROLE  --  every board carries this, any board can use it.
//
//  Plug a USB cable into ANY board and the web page can drive the whole
//  network through it: the tables below are overwritten over USB and then
//  pushed to everyone by radio. Pressing A on a board does the same push from
//  the tables compiled in below, so it also works with no computer attached.
//
//  A board doing this is still a normal neuron -- it keeps its own name,
//  weights and inputs. The two jobs do not overlap.
//
//  NOTE: if you plug USB into two boards at once, both can push. They would
//  push the same data, so it is harmless, but only one is ever needed.
// ============================================================
// ============================================================
//  1. WHICH BOARD IS WHICH NEURON
//
//  Read each board's serial number off the USB serial log (neuron.ts prints
//  it on button B) and paste it in here. The name decides what the board
//  does; the order of this list does not matter.
//
//  Kept as STRINGS on purpose: radio.sendValue() carries a 32-bit FLOAT, exact
//  only for integers up to 2^24 (16777216). Every serial below is far larger,
//  so sending one as a number rounds it (139422584 -> 139422592) and it could
//  never match. Identity therefore travels as text via radio.sendString().
// ============================================================
//  NOTE: the serial is a SIGNED 32-bit int and can be NEGATIVE -- h0 reports
//  -1394225184. Keep the minus sign and every digit exactly as the board prints
//  it. Press B on a board and it prints a ready-to-paste quoted line.
// ONE table for every board -- inputs, hidden neurons and answers alike. The
// two entries line up by position: serialOf("h1") is the serial whose
// allNames[i] is "h1". Looking a serial up BY NAME means nothing depends on a
// second table staying in the same order as this one.
let allNames: string[] = [
    "x0",
    "x1",
    "h0",
    "h1",
    "h2",
    "y0",
    "y1"
]
let allSerials: string[] = [
    "-2022602061",   // -> x0
    "984150378",     // -> x1
    "-1394225184",   // -> h0   confirmed from the board's own dump
    "-1280350870",   // -> h1   <-- UNCONFIRMED, re-read with button B
    "850974008",     // -> h2   <-- UNCONFIRMED, re-read with button B
    "1240653277",    // -> y0   <-- UNCONFIRMED, re-read with button B
    "1843421070"     // -> y1   <-- UNCONFIRMED, re-read with button B
]

// The serial for a board name, or "0" when the page has not set one yet.
function serialOf(name: string): string {
    for (let i = 0; i <= allNames.length - 1; i++) {
        if (allNames[i] == name) {
            return allSerials[i]
        }
    }
    return "0"
}

// Remember which physical board plays `name`, adding it if it is new.
function setSerial(name: string, sn: string) {
    for (let i = 0; i <= allNames.length - 1; i++) {
        if (allNames[i] == name) {
            allSerials[i] = sn
            return
        }
    }
    allNames.push(name)
    allSerials.push(sn)
}

// The neuron boards, in the order their weights arrive: hidden layers first,
// then the answers. Rebuilt by the ARCH command.
let boardNames: string[] = [
    "h0",
    "h1",
    "h2",
    "y0",
    "y1"
]

// ============================================================
//  2. THE NETWORK  --  2-3-2, sigmoid, trained with input scaling OFF
//
//  Straight from the training page's CSV export. One entry per neuron, in
//  order h0, h1, h2, y0, y1; each is [bias, w0, w1, ...] where wJ is the
//  weight on the J-th incoming connection.
//
//  Wire it so those J's line up with the pins:
//      h0/h1/h2 :  P0 = x0,  P1 = x1          (from the two x boards)
//      y0/y1    :  P0 = h0,  P1 = h1,  P2 = h2
// ============================================================
let netBias: number[] = [
    8.468311946492,           // h0
    10.151287262326619,       // h1
    -9.865710916162397,       // h2
    -4.758934173183672,       // y0
    4.758934173183669         // y1
]
let netWeights: number[][] = [
    [-15.777259975468265, 1.6240190707936146],                      // h0 <- x0, x1
    [-11.387501878042364, -0.12801763005923056],                    // h1 <- x0, x1
    [-1.7296569850093202, 11.775940812933898],                      // h2 <- x0, x1
    [-9.282315308018319, 9.533732130267774, 9.517074238087051],     // y0 <- h0, h1, h2
    [10.367666462491574, -8.88285326050701, -9.6635054619619]       // y1 <- h0, h1, h2
]

// ============================================================
//  2d. THE WIRING  --  which neuron feeds which input slot
//
//  This is the table that replaced the cables. One row per neuron, in the same
//  order as netWeights, naming the neurons whose outputs its weights multiply:
//  netSources[k][j] is the source of weight j. The comments on netWeights above
//  already stated this ("y0 <- h0, h1, h2"); with radio it has to be machine
//  readable, because the boards can no longer learn it from how they are wired.
//
//  It MUST line up with netWeights. Swap two names here and the network
//  computes with the weights crossed over -- exactly the failure that swapping
//  two cables used to cause, and just as silent.
//
//  Rebuilt automatically by the ARCH command, so the web page never has to
//  send it: a fully connected network's wiring follows from the layer sizes.
// ============================================================
let netSources: string[][] = [
    ["x0", "x1"],              // h0
    ["x0", "x1"],              // h1
    ["x0", "x1"],              // h2
    ["h0", "h1", "h2"],        // y0
    ["h0", "h1", "h2"]         // y1
]

// ============================================================
//  2a. THE OUTPUT LAYER  --  what each y board announces when it wins
//
//  When an output neuron's value exceeds pushThreshold it flashes its colour
//  and scrolls its message, the way nn.ts did ("frech" / "nicht frech").
//  Edit the text here and it is pushed over radio -- no reflashing.
// ============================================================
let outputNames: string[] = ["y0", "y1"]

// One picture per output neuron, in the same order: a 25-character "0"/"1"
// string, or "" for "no picture, scroll the message instead". Empty by default
// so a board that is sent no artwork behaves exactly as it always did.
let outputPictures: string[] = ["", ""]

// One message per output neuron, in the same order. Used when that neuron has
// no picture.
let outputMessages: string[] = [
    "nicht frech",     // y0 wins
    "frech"            // y1 wins
]

// Flash colour per output neuron, as [red, green, blue], 0-255.
let outputColors: number[][] = [
    [0, 40, 0],        // y0: green
    [40, 0, 0]         // y1: red
]

// The threshold pushed to the answer boards. Named apart from this board's own
// outputThreshold, which is what IT was told to use.
let pushThreshold = 0.5

// The hidden-layer activation pushed to the whole network, matching what the
// training page was set to. Named apart from this board's own actKind.
//   0 sigmoid   1 tanh   2 relu   3 linear
let pushAct = 0

// ============================================================
//  2c. INPUT SCALING
//
//  When the training page standardizes its inputs, the network only works on
//  standardized values. Rather than hide that inside the weights, each x board
//  is told its own mean and spread and does the scaling itself, right before it
//  puts the value on the wire: sent = (chosen - mean) / std.
//  normOn = false means the boards send the raw value.
// ============================================================
let normOn = false
let normMean: number[] = []
let normStd: number[] = []

// ============================================================
//  2b. THE INPUT LAYER  --  what each x board offers to choose from
//
//  Each x board gets a list of selectable values and one picture per value.
//  On the board, A and B step through the list and the picture is displayed;
//  the selected value is sent into the network when a round starts.
//
//  Pictures are 5 rows of 5 characters, "1" = lit, "0" = dark. Edit them here
//  and they are pushed to the boards -- the boards store no pictures of their
//  own. Keep each row exactly 5 characters long.
// ============================================================
let inputNames: string[] = ["x0", "x1"]

// One row per x board: the values that board can send.
let inputChoices: number[][] = [
    [0.25, 0.50, 0.75, 1.00],        // x0
    [0.2, 0.4, 0.6, 0.8, 1.0]        // x1
]

// One picture per choice, in the same order as inputChoices. 5 rows of 5
// characters, "1" = lit. Copied verbatim from the working input.ts, so the
// boards show exactly the artwork they did before -- but sent over radio, so
// changing a picture here needs no reflashing of the neuron boards.
let inputPictures: string[][] = [
    [
        "11011" + "11011" + "00000" + "00000" + "00000",   // x0 = 0.25
        "00011" + "11011" + "00000" + "00000" + "00000",   // x0 = 0.50
        "10010" + "01001" + "00000" + "00000" + "00000",   // x0 = 0.75
        "00000" + "11011" + "00000" + "00000" + "00000"    // x0 = 1.00
    ],
    [
        "00000" + "00000" + "01110" + "10001" + "11111",   // x1 = 0.2
        "00000" + "00000" + "00000" + "10001" + "01110",   // x1 = 0.4
        "00000" + "00000" + "11110" + "10010" + "01101",   // x1 = 0.6
        "00000" + "00000" + "11111" + "11111" + "01110",   // x1 = 0.8
        "00000" + "00000" + "11111" + "10001" + "01110"    // x1 = 1.0
    ]
]

// ============================================================
//  3. Pushing the parameters out
// ============================================================
// Radio messages are small and unacknowledged, so pace them and send the
// identity first -- a neuron ignores weights until it knows its own name.
// Tell a board who it is. Repeated because radio is unacknowledged and this is
// the one message a board cannot do without: miss it and the board ignores
// everything else that follows, staying on "?" for the whole round.
function sendIdentity(name: string) {
    let sn = serialOf(name)
    let msg = sn + "=" + name
    // Print exactly what goes on the air, so it can be compared with the
    // "id? want [...] mine [...]" line each board logs.
    serial.writeLine("id-> [" + msg + "] (from " + sn + ")" +
        (sn == "0" ? "   <-- NO SERIAL SET for " + name : ""))
    for (let attempt = 0; attempt <= 2; attempt++) {
        radio.sendString(msg)
        basic.pause(120)
    }
}

function sendNeuron(index: number) {
    let name = boardNames[index]
    let weights = netWeights[index]

    // Identity first: only the board whose serial number matches takes this
    // name. Everything after it is addressed to the name, so this MUST land --
    // and radio is unacknowledged, so a single send loses the board for the
    // whole round. Sent three times; a repeat is harmless because a board that
    // already has the name simply re-adopts it.
    sendIdentity(name)
    radio.sendValue(name + "i", weights.length)   // fanIn
    basic.pause(60)
    radio.sendValue(name + "b", netBias[index])   // bias
    basic.pause(60)
    for (let j = 0; j <= weights.length - 1; j++) {
        radio.sendValue(name + "w" + j, weights[j])
        basic.pause(60)
    }
    // The wiring: one message per input slot, naming the neuron that feeds it.
    // Without these the board knows its weights but not what to multiply them
    // by, and isConfigured() keeps it unarmed -- so a missing row here shows up
    // as a board that never joins in, not as a wrong answer.
    let srcs = index <= netSources.length - 1 ? netSources[index] : []
    for (let j = 0; j <= weights.length - 1; j++) {
        let src = j <= srcs.length - 1 ? srcs[j] : ""
        if (src == "") {
            serial.writeLine("WARNING " + name + ": no source for input " + j)
            continue
        }
        radio.sendString(name + "s" + j + ":" + src)
        basic.pause(60)
    }
    serial.writeLine("sent " + name + " (" + weights.length + " weights)")
}

// Pack a 25-character "0"/"1" picture into 5 characters, one per row: each row
// is 5 bits (0..31), sent as the character at code 95 + value.
//
// Why: radio.sendString carries at most 19 characters, and
// "<name>p<k>:<25 bits>" is 30 -- it arrived truncated and was silently
// rejected, so the boards only ever showed the fallback dots. Packed, the whole
// message is 10 characters.
//
// Why base 95: codes 95..126 ("_" through "~") are 32 consecutive printable
// characters containing no ":" (the field separator), no quote and no
// backslash, so nothing needs escaping and nothing confuses the parser.
// Pack a MakeCode Image directly, so pictures can be authored in the editor's
// visual grid instead of as "0"/"1" text:
//
//     sendPicture("x0", 0, images.createImage(`
//         . . . . .
//         . # . . #
//         . # . . #
//         . . # # #
//         . . . . .
//         `))
//
// Same wire format as packPicture -- 5 bits per character, one per row.
function packImage(img: Image): string {
    let out = ""
    for (let y = 0; y <= 4; y++) {
        let v = 0
        for (let x = 0; x <= 4; x++) {
            if (img.pixel(x, y)) {
                v = v | (1 << x)
            }
        }
        out = out + String.fromCharCode(95 + v)
    }
    return out
}

// Send one MakeCode Image as choice k's picture for board `name` ("x0"/"x1").
// Fits a single radio message (10 characters).
//
// NOT called from sendAll -- it is here to be called by hand when you want to
// push one picture drawn in the MakeCode editor, e.g.
//     sendPicture("x0", 0, images.createImage(`
//         . # . # .
//         . . . . .
//         # . . . #
//         . # # # .
//         . . . . .
//         `))
function sendPicture(name: string, k: number, img: Image) {
    radio.sendString(name + "p" + k + ":" + packImage(img))
}

function packPicture(bits: string): string {
    let out = ""
    for (let y = 0; y <= 4; y++) {
        let v = 0
        for (let x = 0; x <= 4; x++) {
            if (bits.charAt(y * 5 + x) == "1") {
                v = v | (1 << x)
            }
        }
        out = out + String.fromCharCode(95 + v)
    }
    return out
}

// Give a y board its win message, flash colour and threshold. The message is
// sent in chunks because radio.sendString carries only 19 characters; chunk 0
// replaces whatever was there, later chunks append.
function sendOutput(index: number) {
    let name = outputNames[index]
    let msg = outputMessages[index]
    let col = outputColors[index]

    radio.sendValue(name + "t", pushThreshold)
    basic.pause(120)
    radio.sendValue(name + "cr", col[0])
    basic.pause(120)
    radio.sendValue(name + "cg", col[1])
    basic.pause(120)
    radio.sendValue(name + "cb", col[2])
    basic.pause(120)

    // "<name>m<n>:" is 5 characters while the chunk index is one digit, leaving
    // 14 of the 19-character radio limit. At chunk 10 the header grows to 6, so
    // 13 is used from there on -- a long message would otherwise emit a
    // 20-character string that the radio silently truncates.
    let chunk = 0
    let pos = 0
    while (pos < msg.length) {
        let take = chunk < 10 ? 14 : 13
        radio.sendString(name + "m" + chunk + ":" + msg.substr(pos, take))
        basic.pause(150)
        pos = pos + take
        chunk = chunk + 1
    }
    // The picture, if this neuron has one. Sent after the message so a board that
    // has both ends up preferring the picture, and one character per row keeps it
    // inside the 19-character radio limit.
    let pic = index <= outputPictures.length - 1 ? outputPictures[index] : ""
    if (pic != "" && pic.length == 25) {
        radio.sendString(name + "q0:" + packPicture(pic))
        basic.pause(150)
        serial.writeLine("sent " + name + " picture")
    }
    serial.writeLine("sent " + name + " message: " + msg)
}

// Give an x board its identity, its list of values and one picture per value.
function sendInput(index: number) {
    let name = inputNames[index]
    let values = inputChoices[index]
    let pics = inputPictures[index]

    sendIdentity(name)
    radio.sendValue(name + "n", values.length)        // how many choices
    basic.pause(200)

    // How this board scales its value before sending it on. "s" of 0 means
    // "no scaling"; anything else is the divisor, with "m" the mean.
    if (normOn && index <= normMean.length - 1) {
        radio.sendValue(name + "m", normMean[index])
        basic.pause(150)
        radio.sendValue(name + "s", normStd[index])
        basic.pause(150)
    } else {
        radio.sendValue(name + "s", 0)
        basic.pause(150)
    }

    // VALUES FIRST, all of them, before any picture. The values are what the
    // board needs to work at all; the big picture strings are decoration.
    // Interleaving them meant a picture was in flight while the next value was
    // sent, and the value lost the race.
    for (let k = 0; k <= values.length - 1; k++) {
        radio.sendValue(name + "v" + k, values[k])
        basic.pause(200)
    }
    // Send them once more: radio is unacknowledged, and a dropped value leaves
    // a hole the board cannot fill on its own. A repeat just overwrites its
    // own slot, so it is harmless.
    for (let k = 0; k <= values.length - 1; k++) {
        radio.sendValue(name + "v" + k, values[k])
        basic.pause(200)
    }
    // Pictures last, once the board is already usable. One message each: the
    // 25 bits are PACKED, five per character, so the whole picture fits in the
    // 19-character limit radio.sendString imposes (see packPicture).
    for (let k = 0; k <= pics.length - 1; k++) {
        radio.sendString(name + "p" + k + ":" + packPicture(pics[k]))
        basic.pause(150)
    }
    serial.writeLine("sent " + name + " (" + values.length + " choices)")
}

// Catch a mis-edited table before it goes on the air: a missing picture or a
// wrong-length row would otherwise just silently fail to show on the board.
function checkInputTables() {
    for (let i = 0; i <= inputNames.length - 1; i++) {
        if (inputChoices[i].length != inputPictures[i].length) {
            serial.writeLine("WARNING " + inputNames[i] + ": " +
                inputChoices[i].length + " values but " +
                inputPictures[i].length + " pictures")
        }
        for (let k = 0; k <= inputPictures[i].length - 1; k++) {
            if (inputPictures[i][k].length != 25) {
                serial.writeLine("WARNING " + inputNames[i] + " picture " + k +
                    ": " + inputPictures[i][k].length + " chars, need 25")
            }
        }
        if (serialOf(inputNames[i]) == "0") {
            serial.writeLine("WARNING " + inputNames[i] + ": serial not set")
        }
    }
}

// Configure THIS board straight from the tables, without the radio.
//
// A board never receives its own broadcast, so the board doing the pushing
// would otherwise set everyone up except itself -- it would sit on "?" while
// the rest of the network came alive. The old design never hit this because
// the sender was not a neuron.
function configureSelf() {
    let mySerial = convertToText(control.deviceSerialNumber())
    for (let i = 0; i <= allNames.length - 1; i++) {
        if (allSerials[i] != mySerial) {
            continue
        }
        let nm = allNames[i]
        ownName = nm
        actKind = pushAct
        displayDirty = true
        serial.writeLine("self-configured as " + nm + ", activation " + actName())

        // An x board: values, scaling and pictures.
        for (let k = 0; k <= inputNames.length - 1; k++) {
            if (inputNames[k] != nm) {
                continue
            }
            choiceCount = inputChoices[k].length
            choiceIndex = 0
            choiceValues = []
            choiceValuesSeen = []
            choicePics = []
            for (let v = 0; v <= inputChoices[k].length - 1; v++) {
                choiceValues.push(inputChoices[k][v])
                choiceValuesSeen.push(true)
            }
            for (let pi = 0; pi <= inputPictures[k].length - 1; pi++) {
                choicePics.push(inputPictures[k][pi])
            }
            if (normOn && k <= normMean.length - 1) {
                myMean = normMean[k]
                myStd = normStd[k]
            } else {
                myStd = 0
            }
        }

        // A hidden or answer neuron: fanIn, bias and weights.
        for (let k = 0; k <= boardNames.length - 1; k++) {
            if (boardNames[k] != nm || k > netWeights.length - 1) {
                continue
            }
            fanIn = netWeights[k].length
            fanInSeen = true
            bias = netBias[k]
            biasSeen = true
            for (let j = 0; j <= netWeights[k].length - 1 && j <= 3; j++) {
                ownWeights[j] = netWeights[k][j]
                weightsSeen[j] = true
                // The wiring, from the same table the radio push reads. Without
                // it this board would have weights but no idea whose values
                // they belong to, and isConfigured() would keep it unarmed --
                // the very problem configureSelf() exists to avoid.
                sources[j] = k <= netSources.length - 1 &&
                    j <= netSources[k].length - 1 ? netSources[k][j] : ""
                sourcesSeen = sourcesSeen | (1 << j)
            }
        }

        // An answer neuron: what it announces when it wins.
        for (let k = 0; k <= outputNames.length - 1; k++) {
            if (outputNames[k] != nm) {
                continue
            }
            outputThreshold = pushThreshold
            if (k <= outputMessages.length - 1) {
                winMessage = outputMessages[k]
            }
            if (k <= outputColors.length - 1) {
                winRed = outputColors[k][0]
                winGreen = outputColors[k][1]
                winBlue = outputColors[k][2]
            }
        }
        return
    }
    serial.writeLine("this board's serial " + mySerial + " is not in the table")
}

function sendAll() {
    basic.showString("S")
    checkInputTables()
    // Set ourselves up first: our own broadcast will not reach us.
    configureSelf()
    saveToFlash()
    // Which activation everyone uses. Sent first: it changes what every hidden
    // neuron computes, so it should be in place before any weights arrive.
    radio.sendValue("act", pushAct)
    basic.pause(120)
    for (let i = 0; i <= boardNames.length - 1; i++) {
        sendNeuron(i)
    }
    for (let i = 0; i <= inputNames.length - 1; i++) {
        sendInput(i)
    }
    for (let i = 0; i <= outputNames.length - 1; i++) {
        sendOutput(i)
    }
    basic.showIcon(IconNames.Yes)
    basic.pause(400)
    basic.clearScreen()
}

// ============================================================
//  4. Collecting the answer
// ============================================================
let yOut: number[] = [0, 0]
let ySeen: boolean[] = [false, false]

// Folded into neuron.ts's single radio.onReceivedValue -- see collectAnswer().
// Registering onReceivedValue twice REPLACES the first, so there must only ever
// be one in the merged file.
function collectAnswer(name: string, value: number) {
    // Output neurons report "y0o" / "y1o" when they finish a round.
    if (name == "y0o") {
        yOut[0] = value
        ySeen[0] = true
    }
    if (name == "y1o") {
        yOut[1] = value
        ySeen[1] = true
    }
    if (ySeen[0] && ySeen[1]) {
        // Both logits in: the bigger one wins. Sigmoid is monotonic, so this
        // is the same class the training page's softmax would pick.
        let winner = yOut[0] >= yOut[1] ? 0 : 1
        serial.writeLine("y0=" + yOut[0] + "  y1=" + yOut[1] + "  -> class " + winner)
        basic.showNumber(winner)
        ySeen[0] = false
        ySeen[1] = false
    }
}

// ============================================================
//  6. UART -> RADIO BRIDGE
//
//  The browser (kids.html) opens this board over WebUSB and writes plain text
//  lines. Each line is one command; the board answers with a line so the page
//  can wait for it rather than guessing at timing.
//
//    PING                    -> "OK PING"        is anyone there?
//    ARCH <n0,n1,...>        -> "OK ARCH ..."    layer sizes, e.g. "2,3,2"
//    SERIAL <name> <sn>      -> "OK SERIAL <name>"  which physical board plays
//                                                this neuron, by name,
//                                                e.g. "SERIAL h0 -1394225184"
//    NEURON <i> <b> <w,w,..> -> "OK NEURON <i>"  one neuron's bias + weights
//    MSG <i> <text>          -> "OK MSG <i>"     win message for output i
//    COL <i> <r> <g> <b>     -> "OK COL <i>"     win colour for output i
//    ACT <name>              -> "OK ACT <name>"  hidden activation: sigmoid,
//                                                tanh, relu or linear
//    THRESH <v>              -> "OK THRESH"      win threshold
//    NORM <i> <mean> <std>   -> "OK NORM <i>"    how x board i scales its value
//    NORM off                -> "OK NORM off"   send raw values instead
//    CHOICES <i> <v,v,...>   -> "OK CHOICES <i>" an x board's values
//    PIC <i> <k> <25 bits>   -> "OK PIC <i> <k>" one picture, as 0/1 text
//    YPIC <i> <25 bits|->    -> "OK YPIC <i>"    an answer's picture, "-" clears
//    SEND                    -> "OK SEND"        push everything over radio
//    CALC                    -> "OK CALC"        run one round
//
//  Names are assigned positionally: neuron i takes boardNames[i], so the
//  serial-number table above still decides which physical board is which.
//  A line that does not parse answers "ERR <reason>" and changes nothing.
// ============================================================

// Split "a,b,c" into numbers. Returns an empty array for an empty string.
function parseNumbers(csv: string): number[] {
    let out: number[] = []
    let cur = ""
    for (let i = 0; i <= csv.length - 1; i++) {
        let ch = csv.charAt(i)
        if (ch == ",") {
            if (cur.length > 0) {
                out.push(parseFloat(cur))
            }
            cur = ""
        } else if (ch != " ") {
            cur = cur + ch
        }
    }
    if (cur.length > 0) {
        out.push(parseFloat(cur))
    }
    return out
}

// Split a line on spaces into at most `max` fields. Once max-1 fields are out,
// the rest of the line becomes the final field verbatim -- so a win message may
// contain spaces ("MSG 1 sehr frech heute" keeps all three words).
function splitFields(line: string, max: number): string[] {
    let out: string[] = []
    let cur = ""
    let i = 0
    while (i <= line.length - 1) {
        let ch = line.charAt(i)
        if (ch == " " && out.length < max - 1) {
            if (cur.length > 0) {
                out.push(cur)
                cur = ""
            }
            // Skip any run of spaces.
            while (i + 1 <= line.length - 1 && line.charAt(i + 1) == " ") {
                i = i + 1
            }
        } else {
            cur = cur + ch
        }
        i = i + 1
    }
    if (cur.length > 0) {
        out.push(cur)
    }
    return out
}

// Grow an array of arrays so index i exists.
function ensureRow(arr: number[][], i: number) {
    while (arr.length <= i) {
        arr.push([])
    }
}

function handleLine(line: string) {
    // Strip a trailing CR: browsers commonly send "\r\n".
    let s = line
    while (s.length > 0 && (s.charAt(s.length - 1) == "\r" || s.charAt(s.length - 1) == "\n")) {
        s = s.substr(0, s.length - 1)
    }
    if (s.length == 0) {
        return
    }
    // How many fields to split into depends on the command: MSG's text is the
    // rest of the line and must not be split at its spaces, whereas COL needs
    // five separate numbers. Splitting everything at 5 truncated messages to
    // their first word.
    let cmd = splitFields(s, 2)[0]
    let f = splitFields(s, cmd == "MSG" ? 3 : 5)

    if (cmd == "PING") {
        serial.writeLine("OK PING")
        return
    }

    if (cmd == "ARCH") {
        if (f.length < 2) {
            serial.writeLine("ERR ARCH needs sizes")
            return
        }
        let sizes = parseNumbers(f[1])
        if (sizes.length < 2) {
            serial.writeLine("ERR ARCH needs at least 2 layers")
            return
        }
        // The neuron boards are named h0.. then y0.., so rebuild both name
        // tables from the architecture. The serial table keeps its order, so
        // board i keeps the physical board it always had.
        // Hidden neurons are numbered ACROSS all hidden layers (h0, h1, h2, ...),
        // not restarted per layer -- two layers both starting at h0 would give
        // two boards the same name and they would fight over every message.
        //
        // The wiring is rebuilt at the same time. A fully connected network's
        // sources follow from the layer sizes alone -- every neuron in layer l
        // is fed by every neuron in layer l-1, in order -- so the page never
        // has to send the topology separately, and the sources cannot drift out
        // of step with the names.
        let names: string[] = []
        let srcs: string[][] = []
        // The names of the layer feeding the one being built. Layer 0 is the x
        // boards, which are the first hidden layer's sources.
        let prev: string[] = []
        for (let i = 0; i <= sizes[0] - 1; i++) {
            prev.push("x" + i)
        }
        let hn = 0
        for (let l = 1; l <= sizes.length - 2; l++) {
            let layer: string[] = []
            for (let i = 0; i <= sizes[l] - 1; i++) {
                let nm = "h" + hn
                names.push(nm)
                // A copy per neuron: pushing `prev` itself would give every
                // neuron in the layer the same array, and a later edit to one
                // would change them all.
                let mine: string[] = []
                for (let k = 0; k <= prev.length - 1; k++) {
                    mine.push(prev[k])
                }
                srcs.push(mine)
                layer.push(nm)
                hn = hn + 1
            }
            prev = layer
        }
        let outs: string[] = []
        for (let i = 0; i <= sizes[sizes.length - 1] - 1; i++) {
            outs.push("y" + i)
            names.push("y" + i)
            let mine: string[] = []
            for (let k = 0; k <= prev.length - 1; k++) {
                mine.push(prev[k])
            }
            srcs.push(mine)
        }
        boardNames = names
        outputNames = outs
        netSources = srcs
        // The x boards' names come from the architecture too: a 3-input network
        // needs x0..x2, and leaving inputNames at its compiled-in two would
        // mean the third board was never given its choices.
        let ins: string[] = []
        for (let i = 0; i <= sizes[0] - 1; i++) {
            ins.push("x" + i)
        }
        inputNames = ins
        serial.writeLine("OK ARCH " + names.length + " neurons")
        return
    }

    if (cmd == "SERIAL") {
        // Tell the bridge which physical board plays a given neuron, so the
        // page owns the mapping and nobody has to edit the table above.
        // Addressed by NAME ("x0", "h2", "y1"), which needs no role flag and
        // cannot be desynchronised from a second table.
        if (f.length < 3) {
            serial.writeLine("ERR SERIAL needs name serial")
            return
        }
        setSerial(f[1], f[2])
        serial.writeLine("OK SERIAL " + f[1])
        return
    }

    if (cmd == "NEURON") {
        if (f.length < 4) {
            serial.writeLine("ERR NEURON needs index bias weights")
            return
        }
        let i = parseInt(f[1])
        if (i < 0 || i > 40) {
            serial.writeLine("ERR NEURON bad index")
            return
        }
        while (netBias.length <= i) {
            netBias.push(0)
        }
        ensureRow(netWeights, i)
        netBias[i] = parseFloat(f[2])
        netWeights[i] = parseNumbers(f[3])
        serial.writeLine("OK NEURON " + i)
        return
    }

    if (cmd == "MSG") {
        if (f.length < 3) {
            serial.writeLine("ERR MSG needs index text")
            return
        }
        let i = parseInt(f[1])
        while (outputMessages.length <= i) {
            outputMessages.push("")
        }
        outputMessages[i] = f[2]
        serial.writeLine("OK MSG " + i)
        return
    }

    if (cmd == "COL") {
        if (f.length < 5) {
            serial.writeLine("ERR COL needs index r g b")
            return
        }
        let i = parseInt(f[1])
        ensureRow(outputColors, i)
        outputColors[i] = [parseFloat(f[2]), parseFloat(f[3]), parseFloat(f[4])]
        serial.writeLine("OK COL " + i)
        return
    }

    if (cmd == "ACT") {
        // Which activation the hidden layers use, by name, as the page calls it.
        if (f.length < 2) {
            serial.writeLine("ERR ACT needs a name")
            return
        }
        let a = f[1]
        if (a == "sigmoid") {
            pushAct = 0
        } else if (a == "tanh") {
            pushAct = 1
        } else if (a == "relu") {
            pushAct = 2
        } else if (a == "linear") {
            pushAct = 3
        } else {
            serial.writeLine("ERR ACT unknown: " + a)
            return
        }
        serial.writeLine("OK ACT " + a)
        return
    }

    if (cmd == "THRESH") {
        if (f.length < 2) {
            serial.writeLine("ERR THRESH needs a value")
            return
        }
        pushThreshold = parseFloat(f[1])
        serial.writeLine("OK THRESH")
        return
    }

    if (cmd == "NORM") {
        if (f.length < 2) {
            serial.writeLine("ERR NORM needs index mean std, or off")
            return
        }
        if (f[1] == "off") {
            normOn = false
            serial.writeLine("OK NORM off")
            return
        }
        if (f.length < 4) {
            serial.writeLine("ERR NORM needs index mean std")
            return
        }
        let ni = parseInt(f[1])
        while (normMean.length <= ni) {
            normMean.push(0)
            normStd.push(1)
        }
        normMean[ni] = parseFloat(f[2])
        let sd = parseFloat(f[3])
        normStd[ni] = sd == 0 ? 1 : sd
        normOn = true
        serial.writeLine("OK NORM " + ni)
        return
    }

    if (cmd == "CHOICES") {
        if (f.length < 3) {
            serial.writeLine("ERR CHOICES needs index values")
            return
        }
        let i = parseInt(f[1])
        ensureRow(inputChoices, i)
        inputChoices[i] = parseNumbers(f[2])
        serial.writeLine("OK CHOICES " + i)
        return
    }

    // One picture for an ANSWER neuron. Sent as 25 "0"/"1" characters, same as
    // PIC; an empty string clears it so the board goes back to its message.
    if (cmd == "YPIC") {
        if (f.length < 3) {
            serial.writeLine("ERR YPIC needs index bits")
            return
        }
        let yi = parseInt(f[1])
        let ybits = f[2]
        if (ybits != "-" && ybits.length != 25) {
            serial.writeLine("ERR YPIC needs 25 bits, got " + ybits.length)
            return
        }
        while (outputPictures.length <= yi) {
            outputPictures.push("")
        }
        outputPictures[yi] = ybits == "-" ? "" : ybits
        serial.writeLine("OK YPIC " + yi)
        return
    }

    if (cmd == "PIC") {
        if (f.length < 4) {
            serial.writeLine("ERR PIC needs board choice bits")
            return
        }
        let bi = parseInt(f[1])
        let k = parseInt(f[2])
        let bits = f[3]
        if (bits.length != 25) {
            serial.writeLine("ERR PIC needs 25 bits, got " + bits.length)
            return
        }
        while (inputPictures.length <= bi) {
            inputPictures.push([])
        }
        while (inputPictures[bi].length <= k) {
            inputPictures[bi].push("0000000000000000000000000")
        }
        inputPictures[bi][k] = bits
        serial.writeLine("OK PIC " + bi + " " + k)
        return
    }

    if (cmd == "SEND") {
        sendAll()
        serial.writeLine("OK SEND")
        return
    }

    if (cmd == "CALC") {
        ySeen[0] = false
        ySeen[1] = false
        radio.sendValue("calc", 0)
        serial.writeLine("OK CALC")
        return
    }

    serial.writeLine("ERR unknown command " + cmd)
}

serial.onDataReceived(serial.delimiters(Delimiters.NewLine), function () {
    handleLine(serial.readLine())
})


// ============================================================
//  Button B: dump this board's state to serial (read-only)
// ============================================================
input.onButtonEvent(Button.A, input.buttonEventClick(), function () {
    // x board: step back through the choices.
    if (isInputBoard() && choiceCount > 0) {
        choiceIndex = (choiceIndex + choiceCount - 1) % choiceCount
        showChoice()
        reportChoice()
        return
    }
    // Every other board: print what this neuron knows about itself.
    dumpState(false)
})

// A+B: start a round for the WHOLE network from any board, exactly as the
// sender's B button does. The x boards then send their values, those reach the
// hidden layer over the cables, and the wave runs through to the answers -- so
// a round can be triggered without the sender being plugged in.
//
// This board arms itself too: "calc" is a broadcast, and a board does not hear
// its own radio messages.
input.onButtonEvent(Button.AB, input.buttonEventClick(), function () {
    // x board: start a round for the whole network. The board arms itself too,
    // because a board never hears its own broadcast.
    if (isInputBoard()) {
        serial.writeLine(ownName + ": calc for everyone")
        radio.sendValue("calc", 0)
        armed = true
        return
    }
    // Every other board: push identity, weights, choices and pictures to the
    // whole network -- the old separate sender board's A.
    sendAll()
})

// Everything this board knows about itself, on the USB serial log. Shared by
// button A and button B; only B asks for the full board table as well, so A
// stays short enough to read at a glance.
function dumpState(withTable: boolean) {
    serial.writeLine("--- " + (ownName == "" ? "UNCLAIMED" : ownName) + " ---")
    // Paste this line into the board table on the web page (or allSerials[] in
    // sender.ts), quotes and all.
    // The serial is a SIGNED 32-bit int, so it can be negative -- keep the minus.
    serial.writeLine("    \"" + convertToText(control.deviceSerialNumber()) + "\",")
    basic.pause(50)
    serial.writeLine("serial: " + control.deviceSerialNumber())
    serial.writeValue("fanIn", fanIn)
    serial.writeValue("bias", bias)
    serial.writeLine("configured: " + isConfigured())
    for (let i = 0; i <= 3; i++) {
        serial.writeValue("w" + i, ownWeights[i])
        basic.pause(30)
    }
    for (let i = 0; i <= 3; i++) {
        // The source is printed beside the value: with no cables to trace, this
        // line is the only way to see which neuron a slot is listening to, and
        // a slot showing "(no source)" within fanIn is why a board never fires.
        serial.writeLine("in" + i + ": " + inputs[i] +
            "  received: " + received[i] +
            "  from: " + (sources[i] == "" ? "(no source)" : sources[i]) +
            (i < fanIn ? "" : "   [unused]"))
        basic.pause(30)
    }
    // The whole board table as this board currently holds it. On B only: it is
    // how you check that the serials pushed from the web page actually landed,
    // but it is long, so A does not print it.
    if (withTable) {
        serial.writeLine("--- board table ---")
        for (let t = 0; t <= allNames.length - 1; t++) {
            serial.writeLine("  " + allNames[t] + "  " + allSerials[t] +
                (allSerials[t] == "0" ? "   <-- NOT SET" : ""))
            basic.pause(30)
        }
    }

    // Show the last result again as a bar -- after a round the screen has been
    // cleared, so this is the only way to see what this neuron decided.
    if (hasOutput) {
        serial.writeValue("last_" + ownName, lastOutput)
        led.plotBarGraph(lastOutput, 1)
    } else {
        serial.writeLine(ownName + ": has not calculated anything yet")
        showState()
    }
}

input.onButtonEvent(Button.B, input.buttonEventClick(), function () {
    // x board: step forward through the choices. Elsewhere: dump state.
    if (isInputBoard() && choiceCount > 0) {
        choiceIndex = (choiceIndex + 1) % choiceCount
        showChoice()
        reportChoice()
        return
    }
    dumpState(true)
})

// ============================================================
//  Boot
// ============================================================
// Print this board's serial number the moment it starts, so it can be read off
// without pressing anything -- plug in USB, open the log, and it is there ready
// to paste into the web page's board table. Quoted and comma'd for exactly that.
// Repeated a few times over the first seconds, not printed once: a serial
// monitor is usually opened AFTER the board has already booted, and a single
// line at power-on would be gone before anyone could see it.
control.inBackground(function () {
    for (let boot = 0; boot <= 5; boot++) {
        serial.writeLine("=== " + (ownName == "" ? "unclaimed board" : ownName) +
            " === serial: " + control.deviceSerialNumber())
        serial.writeLine("    \"" + convertToText(control.deviceSerialNumber()) + "\",")
        basic.pause(1500)
    }
})

// Come back as whatever this board was last set up to be, so a power cut does
// not mean re-running the whole setup.
loadFromFlash()

// Show what we are: the restored name, or "?" if nothing was saved.
showState()
