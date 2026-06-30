import { BenchmarkSnapshot } from "../lib/benchmarks";
import { fixed } from "../lib/format";
import SourceBadge from "./SourceBadge";

export default function SpecDecodeTimeline({ snapshot }: { snapshot: BenchmarkSnapshot }) {
  const correctness = snapshot.specDecode.correctness;
  const alpha = snapshot.acceptance.alpha;
  const items = [
    {
      tone: "green",
      title: correctness?.passed ? "Correctness gate PASS" : "Correctness gate unavailable",
      detail: correctness
        ? `distribution-equivalence · n=${correctness.n}`
        : "run bench/run_all.py --correctness"
    },
    {
      tone: "cyan",
      title: `Acceptance settled ${fixed(alpha, 2)}`,
      detail: `${fixed(snapshot.acceptance.minBand, 2)}-${fixed(snapshot.acceptance.maxBand, 2)} benchmark band`
    },
    {
      tone: "purple",
      title: `alpha lift +${fixed(snapshot.specDecode.alphaLiftMean, 2)}`,
      detail: `draft distillation max +${fixed(snapshot.specDecode.alphaLiftMax, 3)}`
    },
    {
      tone: "amber",
      title: "Net throughput 0.6-0.7x",
      detail: "replay tax exceeds the acceptance gain"
    }
  ];

  return (
    <section className="panel timeline-panel">
      <div className="panel-heading">
        <div>
          <h2>Spec decode</h2>
          <p>exact rejection rule plus residual resampling</p>
        </div>
        <SourceBadge kind="benchmark" />
      </div>
      <div className="timeline">
        {items.map((item, index) => (
          <div className="timeline-item" key={item.title}>
            <span className={`timeline-dot ${item.tone}`} aria-hidden="true" />
            {index < items.length - 1 ? <i aria-hidden="true" /> : null}
            <div>
              <strong>{item.title}</strong>
              <p>{item.detail}</p>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

