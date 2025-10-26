# Troubleshooting optional analysis features

This note collects the observed failure modes when running
`scripts/analyze_from_yaml.jl` with the current optional dependencies, together
with concrete steps to resolve each issue.

## 1. BifurcationKit fails to precompile on Julia 1.8

**Symptoms**

```text
ArgumentError: invalid type for argument A in method definition for rmul!
```

This stack trace originates from `BifurcationKit/BorderedArrays.jl` while the
package is loading. The helper in the script reports that the package was found
on disk but could not be imported and that Julia 1.9 is required.

**Cause**

Recent `BifurcationKit.jl` releases adopted method signatures that depend on the
Julia 1.9 type system (e.g. the `StridedMatrix` aliases used in
`rmul!`). When the project is run on Julia 1.8.5, the compiler throws the above
error and the optional dependency check falls back to the internal bifurcation
routines.【F:scripts/analyze_from_yaml.jl†L57-L79】 The project’s manifest was
originally generated on Julia 1.8, so Pkg will still resolve a newer
BifurcationKit version even though it cannot load on that runtime.

**Fixes**

1. **Preferred:** upgrade to Julia ≥ 1.9 (or 1.10). This matches the requirement
   emitted by the script and is compatible with the rest of the project
   dependencies.【F:Project.toml†L26-L29】 After installing the newer Julia,
   re-run `julia --project=. -e 'using Pkg; Pkg.instantiate()'` to rebuild the
   environment.
2. **Alternative:** pin an older BifurcationKit release that still supports
   Julia 1.8.

   ```julia
   julia --project=. -e 'using Pkg; Pkg.add(PackageSpec(name="BifurcationKit", version="0.9"))'
   ```

   Adjust the version to the latest 0.x tag advertised as Julia 1.8 compatible.
   Note that future features in this repository may assume Julia 1.9+, so the
   upgrade path is strongly recommended.

## 2. Attractors.jl is missing

**Symptoms**

The script prints:

```text
Optional dependency Attractors unavailable – falling back
  → Install with: julia --project=. -e 'using Pkg; Pkg.add("Attractors"); Pkg.add("DynamicalSystems")'
```

**Cause**

`Attractors.jl` and its companion `DynamicalSystems.jl` are not listed in the
project’s dependencies, so they are only loaded when the user installs them
explicitly. The optional loader reports the absence and the basin computation
reverts to the self-contained integrator.【F:scripts/analyze_from_yaml.jl†L81-L94】

**Fix**

Install the optional packages inside the project environment:

```julia
julia --project=. -e 'using Pkg; Pkg.add("Attractors"); Pkg.add("DynamicalSystems")'
```

Re-running `scripts/analyze_from_yaml.jl` will now make use of the
Attractors-based basin sampler.

## 3. `Invalid attribute linewidth` when plotting basins

**Symptoms**

During the basin plot step Makie throws:

```text
Invalid attribute linewidth for plot type Poly{…}
```

**Cause**

CairoMakie ≥ 0.12 switched the contour recipe backing type to `Poly`, which no
longer accepts `linewidth`. Earlier versions of the script still passed
`linewidth` when drawing the separatrix overlays, resulting in the runtime
failure you observed.

**Fix**

Update to the current script version, which now renders the contours using the
`strokewidth` attribute that Makie expects for `Poly` plots.【F:scripts/analyze_from_yaml.jl†L1099-L1136】
If you maintain local patches, make sure any custom contour or arrow overlays
also use `strokewidth` instead of `linewidth`.

## 4. Deprecation warnings for phase portrait arrows

**Symptoms**

Makie emits warnings stating that `arrows` and `arrowsize` are deprecated.

**Cause**

`PlottingCairo.phase_portrait!` still uses the legacy `quiver!` recipe, which in
turn dispatches to the deprecated `arrows` helpers.

**Fix**

Switching to `arrows2d!` (as already done for the basin overlays) silences the
warnings and gives finer control over tip geometry. This adjustment is planned
for the plotting utilities; in the meantime the warnings are harmless.

