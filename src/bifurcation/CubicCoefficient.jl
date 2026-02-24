module CubicCoefficient
#=
Spectral computation of the cubic coefficient b(V*) via the
centre-manifold Dirichlet form for the OU-with-resets generator.

Normal form: ȧ = μ(κ) a − b a³
Equilibrium: a*(κ) = sqrt(μ(κ)/b),   β = 0.5

The formula used is:
  b = −⟨ψ, A₁ (−A₀)⁻¹ A₁ φ⟩ / ⟨ψ, φ⟩
where
  A₀  = generator at κ = κ*
  A₁  = ∂_κ A = coupling perturbation operator
  φ   = right leading-odd eigenfunction of A₀
  ψ   = left  leading-odd eigenfunction of A₀

Falls back to the OU approximation b ≈ λ/(3V*) on any failure.
=#

using LinearAlgebra

export compute_b_cubic_spectral

"""
    compute_b_cubic_spectral(V, λ, σ, Θ, c0, ν0; L=6.0, M=401) -> Float64

Return the cubic normal-form coefficient b > 0.
Gracefully returns b = λ/(3V) if spectral computation fails.
"""
function compute_b_cubic_spectral(
    V  ::Float64,          # stationary variance at κ = κ*
    λ  ::Float64,          # OU mean-reversion rate
    σ  ::Float64,          # noise amplitude
    Θ  ::Float64,          # reset threshold
    c0 ::Float64,          # partial-reset factor
    ν0 ::Float64;          # reset Poisson rate
    L  ::Float64 = 6.0,
    M  ::Int     = 401,
)::Float64

    fallback = λ / max(3.0 * V, 1e-8)

    # Grid
    isodd(M) || (M += 1)
    x   = range(-L, L; length = M) |> collect
    h   = step(x)
    σ2h = (σ^2 / 2) / h^2
    w   = fill(h, M); w[1] *= 0.5; w[end] *= 0.5  # trapezoid weights

    # Build FP generator A0 (κ = κ*, so net drift = 0 for leading term)
    A0 = _build_FP(x, h, σ2h, λ, 0.0, Θ, c0, ν0, M)

    # Stationary density ρ₀ (null eigenvector, even symmetry)
    ρ0 = _stationary(A0, w, M)
    ρ0 === nothing && return fallback

    # Leading odd eigenfunction φ and left eigenfunction ψ = −∂_xρ₀
    φ = _leading_odd(A0, x, M)
    φ === nothing && return fallback
    ψ = _minus_deriv(ρ0, h, M)

    # Perturbation: A₁ = −∂_x  (coupling shifts mean; A₁φ = −dφ/dx)
    A1φ = _minus_deriv(φ, h, M)

    # Solve (−A₀) v = A₁φ  with gauge condition w⋅v = 0
    v = _solve_gauge(A0, A1φ, w, M)
    v === nothing && return fallback

    A1v = _minus_deriv(v, h, M)

    num = dot(ψ .* w, A1v)
    den = dot(ψ .* w, φ)
    abs(den) < 1e-14 && return fallback

    b = -num / den
    (isfinite(b) && b > 0) ? b : fallback
end

# -- private helpers ---------------------------------------------------

function _build_FP(x, h, σ2h, λ, κ, Θ, c0, ν0, M)
    A = zeros(M, M)
    for i in 2:M-1
        xi    = x[i]
        drift = -(λ - κ) * xi
        rate  = abs(xi) >= Θ ? ν0 : 0.0
        A[i, i-1] += σ2h - drift / (2h)
        A[i, i]   += -2σ2h - rate
        A[i, i+1] += σ2h + drift / (2h)
        if rate > 0
            xd = c0 * xi
            pos = (xd - x[1]) / h + 1
            j   = clamp(floor(Int, pos), 1, M-1)
            ww  = pos - j
            A[j,     i] += rate * (1 - ww)
            A[j + 1, i] += rate * ww
        end
    end
    A[1, 1] = A[M, M] = -1e6  # absorbing (reflecting boundary approx)
    A
end

function _stationary(A0, w, M)
    A2 = copy(A0)
    b  = zeros(M)
    mid = div(M + 1, 2)
    A2[mid, :] .= w; b[mid] = 1.0
    rho = try A2 \ b catch; return nothing end
    # Enforce even symmetry
    rho .= 0.5 .* (rho .+ reverse(rho))
    rho ./= dot(rho, w)
    rho
end

function _leading_odd(A0, x, M)
    mid = div(M + 1, 2)
    ip  = (mid + 1):M
    n   = length(ip)
    # Restrict A0 to odd subspace: f_odd(x) = f(x) − f(−x)
    Aodd = zeros(n, n)
    for (col, j) in enumerate(ip)
        e         = zeros(M); e[j] = 1.0; e[mid-(j-mid)] = -1.0
        Aodd[:, col] = (A0 * e)[ip]
    end
    eig = try eigen(Aodd) catch; return nothing end
    idx = argmax(real.(eig.values))
    vh  = real.(eig.vectors[:, idx])
    φ   = zeros(M)
    for (k, j) in enumerate(ip)
        φ[j]           =  vh[k]
        φ[mid-(j-mid)] = -vh[k]
    end
    φ ./= maximum(abs.(φ))
    φ
end

function _minus_deriv(f, h, M)
    g = zeros(M)
    for i in 2:M-1; g[i] = -(f[i+1] - f[i-1]) / (2h); end
    g[1] = -(f[2] - f[1]) / h
    g[M] = -(f[M] - f[M-1]) / h
    g
end

function _solve_gauge(A0, rhs, w, M)
    B = -copy(A0)
    mid = div(M + 1, 2)
    B[mid, :] .= w
    r = copy(rhs); r[mid] = 0.0
    try B \ r catch; return nothing end
end

end # module CubicCoefficient
