import OceanBackground from "@/components/ui/OceanBackground";
import OperationalConsole from "@/components/dashboard/OperationalConsole";

export default function Page() {
  return (
    <main className="relative">
      <OceanBackground />

      <section className="relative z-20 px-6 py-20 text-center">
        <div className="mx-auto max-w-3xl">
          <p className="text-sm font-semibold uppercase tracking-[0.3em] text-cyan-300">How it works</p>
          <h1 className="mt-5 text-4xl font-bold leading-tight text-white sm:text-6xl">
            Ocean<span className="text-cyan-300">Embed</span>
          </h1>
          <p className="mx-auto mt-6 max-w-xl text-base leading-relaxed text-white/60 sm:text-lg">
            Satellites see the surface. OceanEmbed reconstructs the North Indian Ocean from 0 to 1000 meters.
          </p>
          <div className="mx-auto mt-12 grid max-w-2xl grid-cols-2 gap-5 sm:grid-cols-4">
            {[
              { val: "0.25°", label: "Grid resolution", desc: "~28 km per cell" },
              { val: "15", label: "Depth levels", desc: "Surface to seafloor" },
              { val: "1 km", label: "Maximum depth", desc: "Full water column" },
              { val: "7 days", label: "Time window", desc: "Rolling context" },
            ].map((s) => (
              <div key={s.label} className="glass rounded-2xl px-5 py-5">
                <p className="font-mono text-2xl font-bold text-white">{s.val}</p>
                <p className="mt-1.5 text-sm font-medium text-cyan-100/80">{s.label}</p>
                <p className="mt-0.5 text-xs text-white/40">{s.desc}</p>
              </div>
            ))}
          </div>
          <a
            href="#console"
            className="mt-12 inline-flex items-center gap-3 rounded-full bg-cyan-400 px-8 py-3.5 text-base font-semibold text-slate-950 shadow-[0_0_28px_rgba(34,211,238,0.35)] hover:bg-cyan-300"
          >
            Explore the Console
          </a>
        </div>
      </section>

      <div className="relative z-20">
        <OperationalConsole />
      </div>
    </main>
  );
}
