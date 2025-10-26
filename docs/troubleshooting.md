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
  → Install with: julia --project=. -e 'using Pkg; Pkg.add("Attractors"); Pkg.add("DynamicalSystemsBase")'
```

**Cause**

`Attractors.jl` and a compatible dynamical systems backend are not listed in the
project’s dependencies, so they are only loaded when the user installs them
explicitly. The optional loader reports the absence and the basin computation
reverts to the self-contained integrator.【F:scripts/analyze_from_yaml.jl†L90-L139】

**Fix**

Install the optional packages inside the project environment:

```julia
julia --project=. -e 'using Pkg; Pkg.add("Attractors"); Pkg.add("DynamicalSystemsBase")'
```

If you prefer the full `DynamicalSystems.jl` stack for additional tooling, it
will be used automatically when present; otherwise the script falls back to the
lightweight `DynamicalSystemsBase.jl` backend to avoid known precompilation
issues on Julia 1.12.【F:scripts/analyze_from_yaml.jl†L90-L139】 Re-running the
analysis will now make use of the Attractors-based basin sampler.

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
`strokewidth` attribute that Makie expects for `Poly` plots.【F:scripts/analyze_from_yaml.jl†L1122-L1175】
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

## 5. `invalid assignment to constant DynamicalSystemsVisualizations.subscript`

**Symptoms**

Adding `Attractors.jl` and `DynamicalSystems.jl` to the project and running
`Pkg.precompile()` (or starting the YAML analysis script) fails with:

```text
ERROR: LoadError: invalid assignment to constant DynamicalSystemsVisualizations.subscript
```

The stack trace points at
`~/.julia/packages/DynamicalSystems/*/ext/DynamicalSystemsVisualizations.jl`.

**Cause**

`DynamicalSystems.jl` bundles the `DynamicalSystemsVisualizations` extension so
that attractor basins can be rendered with Makie. Versions of the extension
prior to `0.6` attempt to reassign the global `subscript` lookup table during
initialisation. Julia 1.12 tightened constant redefinition rules, so the
assignment that previously worked on Julia 1.10/1.11 now throws the above load
error when the package precompiles.【F:scripts/analyze_from_yaml.jl†L60-L77】

**Fixes**

1. Re-run the YAML analysis script. When it encounters the `subscript`
   redefinition it automatically patches the installed
   `DynamicalSystemsVisualizations.jl` extension by rewriting the offending
   assignment as a constant (creating a `.bak` backup the first time). The
   script then retries the load transparently before continuing.【F:scripts/analyze_from_yaml.jl†L60-L139】

2. Update `DynamicalSystems.jl` (and therefore the extension) to a release that
   supports Julia 1.12:

   ```julia
   julia --project=. -e 'using Pkg; Pkg.update(); Pkg.add(PackageSpec(name="DynamicalSystems", version="3.0"))'
   ```

   Any newer release that depends on `DynamicalSystemsVisualizations ≥ 0.6`
   avoids the reassignment. `Pkg.update()` ensures Makie and the visualisation
   stack match the extension requirements.

3. If you prefer a manual workaround, edit
   `~/.julia/packages/DynamicalSystems/*/ext/DynamicalSystemsVisualizations.jl`
   and change the offending line to declare the lookup table as a constant:

   ```julia
   const subscript = Dict("0" => '₀', "1" => '₁', …)
   ```

   Precompile again afterwards. This mirrors the fix applied upstream until you
   can update the package versions.

After any fix, rerun `julia --project=. scripts/analyze_from_yaml.jl …`. The
script will prefer the full `DynamicalSystems.jl` module when it loads cleanly
and otherwise fall back to the patched `DynamicalSystemsBase.jl` backend so
Attractors support remains available.【F:scripts/analyze_from_yaml.jl†L225-L251】

