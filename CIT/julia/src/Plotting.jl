module CITPlotting

using Printf, Statistics
using ..MonteCarlo: ResponseResult
using ..KernelStatistics: KernelSummary, laplace_transform
using ..GeneratorCheck: GeneratorResult

function points(xs, ys, x0, y0, w, h, xmin, xmax, ymin, ymax)
    join((@sprintf("%.2f,%.2f", x0 + w*(x-xmin)/(xmax-xmin), y0 + h*(1-(y-ymin)/(ymax-ymin))) for (x,y) in zip(xs,ys)), " ")
end

function panel!(io, x0, y0, w, h, title, xlabel, ylabel)
    println(io, "<rect x='$x0' y='$y0' width='$w' height='$h' fill='white' stroke='#cccccc'/>")
    println(io, "<text x='$(x0+8)' y='$(y0+20)' font-size='15' font-weight='bold'>$title</text>")
    println(io, "<text x='$(x0+w/2)' y='$(y0+h-5)' text-anchor='middle' font-size='12'>$xlabel</text>")
    println(io, "<text x='$(x0+12)' y='$(y0+h/2)' transform='rotate(-90 $(x0+12) $(y0+h/2))' text-anchor='middle' font-size='12'>$ylabel</text>")
end

function histogram_density(values; bins=80, xmin=-3.3, xmax=3.3)
    counts = zeros(Float64,bins)
    width = (xmax-xmin)/bins
    for value in values
        if xmin <= value <= xmax
            j = clamp(floor(Int,(value-xmin)/width)+1,1,bins)
            counts[j] += 1
        end
    end
    counts ./= (sum(counts)*width)
    centres = [xmin + (j-0.5)*width for j in 1:bins]
    centres, counts
end

function main_figure(response::ResponseResult, summary::KernelSummary, generator::GeneratorResult, output::AbstractString)
    times = response.times
    kernel = summary.mean
    positive = max.(kernel,0.0)
    # Positive mass is retained only for support diagnostics. The cumulative
    # theorem susceptibility and characteristic determinant use the raw kernel.
    raw_increments = vcat(0.0,(kernel[1:end-1] .+ kernel[2:end]) .* diff(times) ./ 2)
    cumulative = cumsum(raw_increments)
    arguments = collect(range(0,2.2; length=280))
    phi = laplace_transform(times,kernel,arguments)
    centres,density = histogram_density(response.stationary_u)

    width,height = 1200,850
    pw,ph = 540,350
    positions = [(60,40),(640,40),(60,450),(640,450)]
    mkpath(dirname(output))
    open(output,"w") do io
        println(io,"<svg xmlns='http://www.w3.org/2000/svg' width='$width' height='$height' viewBox='0 0 $width $height'>")
        println(io,"<rect width='100%' height='100%' fill='white'/>")

        x0,y0 = positions[1]; panel!(io,x0,y0,pw,ph,"(A) Stationary benchmark geometry","Deviation u","Density")
        println(io,"<rect x='$(x0+pw*(2.3/6.6))' y='$(y0+30)' width='$(pw*(2/6.6))' height='$(ph-65)' fill='#9ecae1' opacity='0.18'/>")
        println(io,"<polyline fill='none' stroke='#0b3c5d' stroke-width='2.5' points='$(points(centres,density,x0+35,y0+30,pw-50,ph-65,-3.3,3.3,0,max(maximum(density)*1.08,0.1)))'/>")

        x0,y0 = positions[2]; panel!(io,x0,y0,pw,ph,"(B) Response kernel and support","Time t","k_h(t)")
        println(io,"<polyline fill='none' stroke='#d62728' stroke-width='2.5' points='$(points(times,kernel,x0+35,y0+30,pw-50,ph-65,0,maximum(times),-0.03,1.05))'/>")
        sx = x0+35+(pw-50)*summary.support_997/maximum(times)
        println(io,"<line x1='$sx' y1='$(y0+30)' x2='$sx' y2='$(y0+ph-35)' stroke='#1f77b4' stroke-dasharray='6,5'/>")
        println(io,"<text x='$(x0+pw-220)' y='$(y0+110)' font-size='12'>MC raw Φ(0)=$(@sprintf("%.3f",summary.susceptibility))</text>")
        println(io,"<text x='$(x0+pw-220)' y='$(y0+128)' font-size='12'>Generator Φ(0)=$(@sprintf("%.3f",generator.susceptibility))</text>")

        x0,y0 = positions[3]; panel!(io,x0,y0,pw,ph,"(C) Cumulative raw susceptibility","Time t","Integrated raw response")
        ymax = max(maximum(cumulative),generator.susceptibility)*1.15
        println(io,"<polyline fill='none' stroke='#1f77b4' stroke-width='2.5' points='$(points(times,cumulative,x0+35,y0+30,pw-50,ph-65,0,maximum(times),0,ymax))'/>")
        for (value,color) in ((summary.susceptibility,"#1f77b4"),(generator.susceptibility,"#ff7f0e"))
            yy = y0+30+(ph-65)*(1-value/ymax)
            println(io,"<line x1='$(x0+35)' y1='$yy' x2='$(x0+pw-15)' y2='$yy' stroke='$color' stroke-dasharray='6,5'/>")
        end

        x0,y0 = positions[4]; panel!(io,x0,y0,pw,ph,"(D) Characteristic determinant","Real argument x","1-κΦ_h(x)")
        for (ratio,color) in ((0.75,"#2ca02c"),(1.0,"#ff7f0e"),(1.25,"#9467bd"))
            curve = 1 .- ratio*summary.threshold .* phi
            println(io,"<polyline fill='none' stroke='$color' stroke-width='2.5' points='$(points(arguments,curve,x0+35,y0+30,pw-50,ph-65,0,2.2,-0.55,0.75))'/>")
            println(io,"<text x='$(x0+pw-125)' y='$(y0+55+round(Int,35*(ratio-0.75)/0.25))' font-size='12' fill='$color'>κ=$(ratio)κ*</text>")
        end
        zero_y = y0+30+(ph-65)*(1-(0+0.55)/(0.75+0.55))
        println(io,"<line x1='$(x0+35)' y1='$zero_y' x2='$(x0+pw-15)' y2='$zero_y' stroke='#555555'/>")
        println(io,"</svg>")
    end
    output
end

end
