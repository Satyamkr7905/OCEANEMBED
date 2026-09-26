"use client";

export default function HeaderBar({
  date,
  onDate,
  online,
}: {
  date: string;
  onDate: (d: string) => void;
  online: boolean;
}) {
  return (
    <header className="sticky top-0 z-30 border-b border-white/10 bg-black/25 backdrop-blur-2xl">
      <div className="px-6 py-4">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-4">
            <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-cyan-400 text-sm font-bold text-slate-950 shadow-[0_0_18px_rgba(34,211,238,0.45)]">
              OE
            </div>
            <div>
              <h2 className="text-lg font-bold text-white">OceanEmbed Console</h2>
              <p className="text-xs font-medium text-cyan-200/70">INCOIS maritime intelligence</p>
            </div>
          </div>

          <div className="flex items-center gap-5">
            <div className="flex items-center gap-3">
              <span className="text-sm font-medium text-white/60">Date</span>
              <input
                type="date"
                value={date}
                onChange={(e) => onDate(e.target.value)}
                className="rounded-xl border border-white/15 bg-white/10 px-3 py-2 text-sm font-medium text-white outline-none transition-shadow focus:border-cyan-300 focus:shadow-glow [color-scheme:dark]"
              />
            </div>
            <div
              className={`flex items-center gap-2 rounded-full px-4 py-2 text-sm font-semibold ${
                online ? "bg-teal-400/15 text-teal-200" : "bg-orange-400/15 text-orange-200"
              }`}
            >
              <span className={`inline-block h-2.5 w-2.5 rounded-full ${online ? "animate-pulse bg-teal-300" : "bg-orange-400"}`} />
              {online ? "Live" : "Offline"}
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}
