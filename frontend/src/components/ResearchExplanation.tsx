"use client";

type Patch = {
  x: number;
  y: number;
  width: number;
  height: number;
  delta_logit: number;
};

export type ResearchExplanationPayload = {
  status: "ok" | "unavailable";
  reason?: string;
  research_only?: boolean;
  publication_decision?: string;
  independent_authentication?: boolean;
  classification?: { decision: string; model: string; model_version: string };
  sensitivity?: {
    method: { name: string; version: string; weak_abs_delta?: number };
    model_id: string;
    model_version: string;
    preprocessing_id: string;
    baseline_logit: number;
    baseline_decision: string;
    max_abs_delta_logit: number;
    patches: Patch[];
    evidence_supports_visual_summary: boolean;
  } | null;
  explanation?: {
    observation: string;
    hypothesis: string;
    limitations: string;
    source: string;
  } | null;
};

export function ResearchExplanation({
  imageUrl,
  imageWidth,
  imageHeight,
  payload,
  pending,
}: {
  imageUrl: string | null;
  imageWidth: number;
  imageHeight: number;
  payload: ResearchExplanationPayload | null;
  pending: boolean;
}) {
  const sensitivity = payload?.status === "ok" ? payload.sensitivity : null;
  const maxDelta = sensitivity?.max_abs_delta_logit ?? 0;
  return (
    <section className="space-y-3 rounded-xl border border-primary/15 p-4" aria-label="What influenced this model classification?">
      <h2 className="text-base font-semibold text-primary">What influenced this model classification?</h2>
      <p className="text-sm font-semibold text-primary">
        Research only. This explanation is not independent authentication.
      </p>
      {pending ? <p className="text-sm text-primary/70">Measuring which masked regions change the model score.</p> : null}
      {payload?.status === "unavailable" ? (
        <p className="text-sm text-primary/70" role="status">
          Explanation unavailable. {payload.reason} The classification above is unchanged.
        </p>
      ) : null}
      {sensitivity && imageUrl ? (
        <div className="relative inline-block max-w-full">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={imageUrl} alt="Research image with a sensitivity overlay" className="max-h-80 rounded-lg" />
          <div className="pointer-events-none absolute inset-0">
            {sensitivity.patches.map((patch) => {
              const threshold = sensitivity.method.weak_abs_delta ?? 0.001;
              if (Math.abs(patch.delta_logit) < threshold || maxDelta <= 0) return null;
              const strength = Math.abs(patch.delta_logit) / maxDelta;
              return (
                <div
                  key={`${patch.x}-${patch.y}`}
                  style={{
                    position: "absolute",
                    left: `${(patch.x / imageWidth) * 100}%`,
                    top: `${(patch.y / imageHeight) * 100}%`,
                    width: `${(patch.width / imageWidth) * 100}%`,
                    height: `${(patch.height / imageHeight) * 100}%`,
                    background: `rgba(255, 122, 26, ${0.15 + strength * 0.55})`,
                  }}
                />
              );
            })}
          </div>
        </div>
      ) : null}
      {payload?.explanation ? (
        <div className="space-y-2 text-sm text-primary/80">
          <p>{payload.explanation.observation}</p>
          <p>{payload.explanation.hypothesis}</p>
          <p className="font-semibold text-primary">What this does not tell us</p>
          <p>{payload.explanation.limitations}</p>
          <p className="text-xs text-primary/55">
            Method {sensitivity?.method.name} v{sensitivity?.method.version}. Model {sensitivity?.model_id} version{" "}
            {sensitivity?.model_version}. Preprocessing {sensitivity?.preprocessing_id}. Language source{" "}
            {payload.explanation.source}. Publication remains {payload.publication_decision}.
          </p>
        </div>
      ) : null}
    </section>
  );
}
