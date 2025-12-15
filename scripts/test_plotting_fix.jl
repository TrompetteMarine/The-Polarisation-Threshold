#!/usr/bin/env julia
"""
Test minimal plotting with fixed contourf syntax
"""

using Plots

# Create simple test data
x = range(0.0, 1.0; length=10)
y = range(0.0, 1.0; length=10)
z = [sin(3*xi) * cos(3*yi) for yi in y, xi in x]

# Compute color limits
z_finite = filter(isfinite, vec(z))
clims = (minimum(z_finite), maximum(z_finite))

# Create levels
levels = collect(range(clims[1], clims[2]; length=15))

println("Testing contourf with fixed syntax...")
println("  clims = $clims")
println("  levels: $(length(levels)) points")

# Test plot
plt = plot(size=(800, 600), dpi=150)

# This should work with the fixes
contourf!(plt, x, y, z;
          levels=levels,
          c=:viridis,
          clims=(clims[1], clims[2]),
          xlabel="x", ylabel="y",
          title="Test Plot",
          colorbar=true)

contour!(plt, x, y, z;
         levels=levels,
         color=:black,
         linewidth=0.5,
         label=false,
         colorbar=false)

mkpath("figs")
savefig(plt, "figs/test_plot.png")

println("✓ Plotting test successful!")
println("  Saved: figs/test_plot.png")
