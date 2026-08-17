# Geometry worker tuning

VibeCAD tunes geometry inspection against deterministic CAD truth first and
language-model behavior second. A model can reveal ambiguity, but it cannot
decide whether a measured radius, bound, selector, or topology count is
correct.

## Local deterministic benchmark

Build `VibeCADGeometryWorker`, then run:

```bash
cmake --build build/release --target VibeCADGeometryWorker
python3 tools/geometry_worker_benchmark.py --repeat 3
```

The Python entry point locates the local release `FreeCADCmd` and relaunches
itself in that runtime. Set `VIBECAD_FREECADCMD` to test another build.

Add the repository BREP regression fixtures and retain the complete report:

```bash
python3 tools/geometry_worker_benchmark.py \
    --repeat 3 \
    --repository-breps \
    --output geometry-worker-results.json
```

The analytic corpus has independent expectations for volume, bounds,
topology, analytic surface queries, axes, and radii. The repository corpus
compares serialized worker results with exact in-process OCC topology and
`optimalBoundingBox(False, False)`. Cached triangulation is never accepted as
geometric truth.

Every run records:

- exact expectation failures;
- response stability after timing fields are removed;
- worker and end-to-end latency;
- compact response size;
- query cardinality;
- the slowest case.

## Model-use loop

After the deterministic corpus is green, exercise the same build through the
real VibeCAD provider session. Local `qwen3.5:9b` is the clarity canary because
small models expose vague names, redundant data, and underspecified choices
quickly. A stronger model is the generalization check.

Model feedback is diagnostic evidence only. Accept a contract or
implementation change when it:

1. fixes a reproducible geometry or interaction failure;
2. preserves or improves the deterministic corpus;
3. reduces wrong or recovery tool calls on the same task;
4. does not introduce task-specific instructions or forced tool use;
5. remains clear to a stronger model.

Record model, allocated context, CPU/GPU split, prompt, tool trace, final
artifact, final answer, and wall time. Keep model inference time separate from
worker time.

## CADGenBench

[CADGenBench](https://github.com/huggingface/cadgenbench) provides realistic
generation and editing tasks. Its
[public inputs](https://huggingface.co/datasets/HuggingAI4Engineering/cadgenbench-data)
contain 49 drawing-based generation tasks and 32 STEP editing tasks. The
private leaderboard truth remains the final shape-quality authority.

Use its editing STEP files as inspection workloads after the local analytic
and repository corpora pass. Do not copy private or downloaded benchmark data
into this repository. Candidate STEP files can be submitted to the official
leaderboard when end-to-end VibeCAD generation is being evaluated.

## Initial baseline

The first tuning iteration found that tessellation deflection inflated a
12 mm major-radius, 3 mm minor-radius torus from its exact 30 × 30 × 6 mm
bounds to approximately 32.47 × 32.47 × 6 mm. Exact OCC bounds corrected the
result.

The corrected worker passed all 35 initial cases. Median worker time was about
51 ms, p95 was about 252 ms, and the 996-face/2,115-edge stress case completed
in about 853 ms. A live 65K-context `qwen3.5:9b` VibeCAD turn then created the
torus, called `vibescript.read_geometry`, and reported the corrected measured
30 × 30 × 6 mm dimensions.

## Strong-model generalization check

A live ChatGPT-subscription `gpt-5.6-terra` turn at high reasoning repeated the
same torus task. It made one creation call, waited for that asynchronous
operation with two status reads, independently inspected the accepted object,
and captured the viewport. The complete turn took 27.9 seconds. The comparable
Qwen turn took 435.2 seconds. This is a single-task comparison, not a general
model-quality claim.

The first Terra run also exposed a VibeCAD defect: the creation result used a
cached, tessellation-inflated `Shape.BoundBox`, while `read_geometry` returned
exact OCC bounds. Qwen had received the same contradictory results. Output
publication now uses the same exact OCC bound calculation as inspection. A
second live Terra run confirmed that both calls report
30.0000002 × 30.0000002 × 6.0000002 mm for the torus.

Exact output bounds have a measurable cost. On the 996-face/2,115-edge BREP,
publishing shape facts took about 1.08 seconds instead of roughly 6 ms for the
cached native bound. This work runs in the isolated geometry process; the
accuracy cost must still be watched on larger imported models.

Both live GUI runs produced and saved the expected artifacts before the known
headless Qt teardown crash. That teardown failure is separate from geometry
creation and inspection, but it means these runs do not establish clean GUI
process shutdown.

A second Terra task created a hollow shaft with two unequal flanges and a
through-bore. In one creation attempt and one inspection call, it found the
four requested cylindrical surfaces by their 4, 10, 14, and 18 mm radii. The
first run took 46.8 seconds and returned correct 36 × 36 × 50 mm bounds.
The model unnecessarily requested both named query matches and the complete
unfiltered face/edge list, making that inspection result about 16.2 KB. The
tool description now states that named query matches do not require
`include_subelements`.

An identical rerun still requested unfiltered details and interpreted the
ambiguous phrase "40 mm long main shaft" differently, as a 40 mm overall
length. It remained valid and inspected its geometry correctly, but it did not
demonstrate that the wording change reduced model behavior noise. Keep this
distinction: the geometry contract was correct in both runs; natural tool
selection and ambiguous design intent remain model/prompt variables.
