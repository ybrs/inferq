# What else can the small models do?

[`report.md`](report.md) and [`guards.md`](guards.md) measure one job — organizing
and summarizing text without inventing things. This file answers four separate
questions about widening the scout's remit, mostly about Qwen3-0.6B, whose
~50 t/s decode and ~356 t/s prefill make it tempting for jobs a 3B would make
you wait for.

**Verdict up front:**

| Job | Qwen3-0.6B | Qwen3-1.7B | granite-4.1-3b |
| --- | --- | --- | --- |
| Dutch → English | fluent and **wrong** — flips dates, invents nouns | mostly right, misses idiom | reliable |
| "What is this file about?" | works | works | works |
| "Which files should I open?" | right neighbourhood, sloppy list | **invents filenames** | right, under-fills the list |
| Workflow → JSON DAG | valid JSON, **serialises parallel branches** | invalid JSON | correct |
| Working Python script | no | no | yes |
| Mermaid diagram | degenerates | no edges at all | valid, but wrong topology |
| matplotlib chart | renders a **misleading** chart | crashes | crashes |

Mechanically-scored probes (`grade_cap.py`; the translation and file-summary
probes need a reader and are discussed below):

| Model | DAG | script | mermaid | chart | route 1 | route 2 | total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| granite-4.1-3b Q4_K_M | 10/10 | 8/8 | 6/8 | 3/6 | 4/4 | 3/4 | **34/40** |
| Qwen3-0.6B Q4_K_M | 8/10 | 4/8 | 0/8 | 4/6 | 4/4 | 3/4 | 23/40 |
| Qwen3-1.7B Q4_K_M | 0/10 | 4/8 | 2/8 | 3/6 | 3/4 | 4/4 | 16/40 |

## 1. Dutch → English

**Don't.** Not with the 0.6B. Its English reads well, which is the problem —
nothing in the output signals that a fact changed. From one support ticket
(`x1`), against the Dutch source:

| Dutch | Correct | Qwen3-0.6B produced |
| --- | --- | --- |
| bij het opvragen van zendingen | when retrieving shipments | "when processing incoming messages" |
| Zendingnummer ZND-88213 faalt reproduceerbaar | shipment ZND-88213 fails reproducibly | "Sending number ZND-88213 reproduces" |
| Kunnen jullie **eventueel vandaag** nog terugkoppelen? | could you possibly get back to us **today**? | "Can you please try to return to the system **tomorrow**?" |

An urgency request for *today* came out as *tomorrow*, and the subject of the
whole ticket changed from shipments to messages. The delivery ID survived, which
is exactly what makes it dangerous: the parts a human spot-checks look right.

The false-friend probe (`x3`) separates the three cleanly. Dutch `actueel` means
*up to date*, `begroting` is *budget*, `magazijnmeester` is *warehouse manager*,
and `dat scheelt een hoop gedoe` means *that saves a lot of hassle*:

- **0.6B** — "not relevant", "the offer" (for *budget*), "the **magazine
  editor**", and "that's **a hope gone**". Four semantic errors in five sentences,
  one of them inverting the meaning.
- **1.7B** — up to date ✓, budget ✓, warehouse manager ✓, but "that's a bit of a
  mess" for *that saves a lot of hassle* — still an inversion.
- **granite-4.1-3b** — all four right, including *request a new quote* for
  `offerte opvragen`.

On the technical paragraph (`x2`) the same ordering holds: granite renders
*doorvoer* as **throughput** and *uitpakken* as **unpacking**; the 1.7B says
"transmission" and "unrolling"; the 0.6B opens with "The decoder of the decoding
path", drops "on this machine", and turns *35 percent lower* into "35 percentage
points". If a translation feeds anything downstream, use granite.

## 2. Reading code, and picking files to open

This is the use the 0.6B is actually suited for.

**"What is this file about" works, even at 0.6B.** Given `src/sampling.rs` with
no docstring to lean on, it answered: *"a Rust implementation of a sampling
configuration and a sampler for text generation… defines the parameters for the
sampling process and the logic for generating samples from a given set of
logits."* Correct, and short. All three models passed this probe. It is a good
indexer: one file in, one line out, ~1–3s at this size.

Watch for embroidery. On `python/reference_logits.py` the 0.6B correctly
identified the reference-logits CLI and then added that it compares "Rust and
**C++** code" — there is no C++ anywhere near it. The one-liner is trustworthy;
the clause after it may not be.

**File routing is usable, and its failures are cheap to catch.** Given a listing
of twelve `src/*.rs` files and a question, asked for the three most likely:

| Question | 0.6B | 1.7B | granite |
| --- | --- | --- | --- |
| Where is the KV cache allocated? | config, loader, **runtime** ✓ | **`src/kv_cache.rs`** (does not exist), runtime, config | gguf, **runtime** ✓, qgemm |
| Tool-call JSON comes back mangled | **tool_calls** ✓, sampling, tool_calls *(duplicate)* | **tool_calls** ✓, trace, runtime | **tool_calls** ✓ only — one pick, not three |

Every model found the right file. The interesting column is how they fail: the
1.7B **invented a path** that fits the naming convention perfectly, the 0.6B
wasted a slot on a duplicate, and granite ignored the count. All three failures
are detectable in five lines of caller code — intersect the answer with the real
file list, dedupe, and top up if short. That is what makes this job safe for a
small model: a wrong pick costs one wasted `read`, and a fabricated path cannot
even be opened.

So the "context pre-pass" idea holds up. The caution is the same one from
[`guards.md`](guards.md): the 0.6B under-segments. Ask it about a question with
three parts and it will answer for the part it noticed. Split compound questions
before handing them over, one query at a time.

## 3. Workflows and DAGs

**Structure: yes. Parallelism: no.** The probe describes seven steps where two
pulls feed one validation, validation feeds two independent loads, and both
loads feed a rebuild and then an email — seven nodes, seven edges, two places
where work fans out.

granite got it exactly right, all seven edges. Qwen3-1.7B emitted **invalid
JSON** (mismatched brackets), and its topology was a straight chain. Qwen3-0.6B
emitted clean, valid JSON with all seven nodes — and then wired the two
independent loads in series:

```json
"edges": [["raw_orders_export","validation"], ["validation","load_orders"],
          ["load_orders","load_returns"], ["load_returns","daily_summary"],
          ["daily_summary","summary_email"]]
```

Five edges instead of seven. `returns_export` is declared and then never
connected to anything, and `load_returns` now waits on `load_orders` for no
reason. As a workflow engine input this is the bad kind of wrong: it is valid,
it runs, and it silently runs serially with a dependency nobody asked for, while
one branch never executes at all.

This is the same defect as the dropped task in `c2` — the model does not hold
"two independent things" in mind; it emits a sequence. If you want a small model
in this loop, have it produce the node list only and derive edges yourself, or
make each fan-out an explicit separate question.

**Writing runnable Python: no, and this is settled.** On a CSV-aggregation
script neither small model produced working code. The 0.6B forgot `import sys`,
summed the running count instead of the amount column, and printed one garbled
line; the 1.7B ran and printed `closed: 4.00 orders` — it put the *sum* where the
*count* belongs. granite scored 8/8, output byte-correct including the
missing-column path exiting 1 with a message on stderr. This matches the main
report's `t3` result; nothing about the newer probes changes it.

## 4. Diagrams

**Mermaid: no.** The 0.6B degenerated — it emitted `flowchart TD` and then the
bare word `flowchart` thirty-eight more times, to the token cap. The 1.7B wrote
seven lines like `a.Pull raw orders export` with **no arrows at all** — a list
formatted as a diagram. granite produced syntactically valid Mermaid with real
edges, decision nodes and error branches, but chained the two independent loads
(`F --> H`) and invented a failure-handling path and a cycle back to the start
that the description never mentions. Even at 3B the diagram lost the parallelism
that the same model got right in JSON one probe earlier. **If you want a
diagram, generate the DAG as JSON and render it yourself** — the structured
format is where these models are accurate.

**Chart code: no, for all three.** Asked for a grouped bar chart with a log axis
saved to `sys.argv[1]`, granite and the 1.7B both crashed — a `NameError` on a
misspelled variable and on an undefined `args` respectively. granite's data
mapping was also scrambled, pairing each model with the wrong numbers.

The 0.6B's script is the one worth looking at, because it *runs*:

- it hardcodes `output.png` and ignores the path argument, and
- it calls `plt.bar` twice at the same x positions, so the bars **overlay**
  rather than group. Decode is drawn on top of prefill and reads as a segment of
  it, and there are no axis labels.

A chart that renders and misleads is worse than one that crashes. granite's
failures are single typos that an execute-and-repair loop would fix in one round;
the 0.6B's are semantic and would survive any number of rounds, because nothing
errors.

## What this means for tiering

Of the four, **only code indexing and file routing survive at 0.6B**, and those
are exactly the jobs where a wrong answer costs one wasted file read and can be
validated against ground truth the caller already has. Translation, DAG
authoring, script writing, and diagram generation all fail in the same
characteristic way — plausible, well-formed output with the meaning quietly
changed — and none of them has a cheap validator.

The dividing line is not difficulty, it is **whether the caller can check the
answer**. Route to the 0.6B when you can verify the output mechanically; route
past it when you cannot.

## Reproducing

```bash
cd harness
OUTDIR=$PWD/outputs-cap ./quality.sh Qwen3-0.6B-Q4_K_M.gguf cap
MPL_PYTHON=/path/to/venv/bin/python python3 grade_cap.py outputs-cap
```

The chart probe needs an interpreter with matplotlib installed; `grade_cap.py`
says so explicitly rather than scoring a zero if it is missing. Generated code
runs with its working directory inside a temp dir — the 0.6B's script hardcodes
`output.png`, and on the first run it wrote that into the harness directory. Prompts are in
`harness/tests-capability/`, raw answers in `harness/outputs-cap/`.
