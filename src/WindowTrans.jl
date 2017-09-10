module WindowTrans

using DSP

# The windowed transforms work by sliding a window over the signal and
# performing the Fourier transform. The input signal is assumed to be
# apodized (zero at the ends of the interval).

function wind_transf(t, x, w)
    f = fftshift(fftfreq(length(t),1.0/(t[2]-t[1])))

    Cₓ = hcat([x.*circshift(w,i) for i = 1:length(t)]...)
    Wₓ = fftshift(fft(Cₓ, 1), 1)
    f,Wₓ
end

#=
The Gabor transform
$$
G_x(t,f) = \int_{-\infty}^\infty \mathrm{d}\tau
\exp[-\pi(\tau-t)^2]\exp(-j\pi f\tau)x(\tau)
$$

Here, we use a version with variable width.
=#

gabor(t, x, σ) = wind_transf(t, x, fftshift(exp.(-(t-mean(t)).^2/2σ^2)))

#=
The Wigner distribution function (WDF) is defined as [[1]]

$$
W_x(t,f) =
\int_{-\infty}^\infty d\tau
x\left(t+\frac{\tau}{2}\right)
x^*\left(t-\frac{\tau}{2}\right)
\exp(-j2\pi f\tau)
$$

Thus, for every $t$, we need to form

$$C_x\left(t+\frac{\tau}{2}, t-\frac{\tau}{2}\right)
\equiv x\left(t+\frac{\tau}{2}\right)
x^*\left(t-\frac{\tau}{2}\right),$$

but since the integration limits are $\pm\infty$, we instead form

$$x\left(t\right)
x^*\left(t-\tau\right).$$

The function is assumed to be windowed on input.

The good thing about the WDF is that the resolution is very high. The
bad thing is the cross term that shows up as soon as there are more
than one frequency present in the signal.

[1]: http://dx.doi.org/10.1109/TSP.2007.896271
=#
wigner(t, x) = wind_transf(t, x, fftshift(conj(reverse(x))))

#=
The "NMR" transform uses a window function that is derivating--apodizing:

$$
W(t) = t^n\exp(-\lambda_w t)
$$
=#
nmr(t, x, λ, n = 2) = wind_transf(t, x, t.^n.*exp(-λ*t))

export wind_transf, gabor, wigner, nmr

end # module
