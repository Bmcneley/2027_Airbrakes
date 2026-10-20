"""
Placeholder airbrake simulation for the CyLaunch subscale.

Uses the same rocket as plot_flight.py (built from parameters.json) and adds a
RocketPy air-brake system with a simple apogee-targeting controller.

    python airbrakes_sim.py                       # target from AIRBRAKE config below
    python airbrakes_sim.py --target 750          # override target apogee (m AGL)
    python airbrakes_sim.py --no-show             # just save airbrakes_results.png

Everything marked PLACEHOLDER is a guess -- swap in real numbers once the
hardware (flap size, servo speed, Cd from CFD/wind tunnel/flight data) exists.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rocketpy import Flight

from plot_flight import build_rocket, fn_array

# ----------------------------------------------------------------------------
# PLACEHOLDER airbrake + controller settings
# ----------------------------------------------------------------------------
AIRBRAKE = {
    "target_apogee_m": 800.0,     # PLACEHOLDER mission target (AGL)
    "n_flaps": 3,                 # PLACEHOLDER
    "flap_width_m": 0.025,        # PLACEHOLDER 2.5 cm wide
    "flap_extension_m": 0.03,     # PLACEHOLDER 3 cm out of the body at full deploy
    "cd_full": 1.2,               # PLACEHOLDER flat-plate-ish Cd, referenced to flap area
    "servo_full_travel_s": 0.4,   # PLACEHOLDER time to go 0 -> 100 %
    "controller_hz": 20,          # how often the flight computer updates
    "deploy_delay_after_burnout_s": 0.5,  # rules usually forbid deploying under thrust
}

G = 9.80665


def flap_area(cfg):
    return cfg["n_flaps"] * cfg["flap_width_m"] * cfg["flap_extension_m"]


def brake_cd(deployment, mach):
    """Cd of the brakes (referenced to full flap area) vs deployment & Mach.
    PLACEHOLDER: linear in deployment, no Mach effect."""
    return AIRBRAKE["cd_full"] * deployment


# ----------------------------------------------------------------------------
# Apogee predictor + controller
# ----------------------------------------------------------------------------


def make_controller(env, rocket, drag_clean, cfg, log):
    burnout = rocket.motor.burn_out_time
    a_body = np.pi * rocket.radius**2
    a_brake = flap_area(cfg)
    max_rate = 1.0 / cfg["servo_full_travel_s"]

    # Lookup tables so the in-flight predictor is fast (like a flight computer would)
    h_grid = np.linspace(0, 4000, 401)
    rho_tab = np.array([env.density(h + env.elevation) for h in h_grid])
    sos_tab = np.array([env.speed_of_sound(h + env.elevation) for h in h_grid])
    mach_tab, cd_tab = drag_clean[:, 0], drag_clean[:, 1]

    def predict_apogee(t, z, vz, deployment, dt=0.05):
        """Integrate a 1-D vertical coast with altitude-varying density and Cd(Mach),
        holding the brake deployment constant. Returns predicted apogee (m AGL)."""
        m = rocket.total_mass(t)
        while vz > 0:
            rho = np.interp(z, h_grid, rho_tab)
            mach = vz / np.interp(z, h_grid, sos_tab)
            cda = np.interp(mach, mach_tab, cd_tab) * a_body + brake_cd(deployment, mach) * a_brake
            a = -G - 0.5 * rho * cda * vz * abs(vz) / m
            if vz + a * dt <= 0:                    # last partial step
                return z + vz * vz / (-2 * a)
            z += vz * dt + 0.5 * a * dt * dt
            vz += a * dt
        return z

    def required_deployment(t, z, vz, target):
        """Bisection: constant deployment that makes the predicted apogee hit target."""
        if predict_apogee(t, z, vz, 0.0) <= target:
            return 0.0
        if predict_apogee(t, z, vz, 1.0) >= target:
            return 1.0
        lo, hi = 0.0, 1.0
        for _ in range(10):
            mid = 0.5 * (lo + hi)
            if predict_apogee(t, z, vz, mid) > target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def controller(time, sampling_rate, state, state_history, observed_variables, air_brakes):
        z = state[2] - env.elevation
        vz = state[5]
        dt = 1.0 / sampling_rate
        dep = air_brakes.deployment_level

        if time < burnout + cfg["deploy_delay_after_burnout_s"]:
            new_dep = 0.0
        elif vz <= 0:                          # past apogee: retract
            new_dep = 0.0
        else:
            want = required_deployment(time, z, vz, cfg["target_apogee_m"])
            # servo can only move so fast
            new_dep = float(np.clip(want, dep - max_rate * dt, dep + max_rate * dt))
        pred = predict_apogee(time, z, vz, new_dep) if vz > 0 else z

        air_brakes.deployment_level = new_dep
        log.append((time, z, vz, new_dep, pred))
        return time, new_dep, pred

    return controller


# ----------------------------------------------------------------------------
# Simulation runs
# ----------------------------------------------------------------------------


def fly(params, base_dir, cfg, mode):
    """mode: 'none' (no brakes), 'control', or a float = fixed deployment level."""
    env, rocket, _, _, _, drag_clean = build_rocket(params, base_dir, verbose=False)
    log = []
    if mode != "none":
        if mode == "control":
            ctrl = make_controller(env, rocket, drag_clean, cfg, log)
        else:
            fixed = float(mode)
            burnout = rocket.motor.burn_out_time

            def ctrl(time, sampling_rate, state, state_history, obs, air_brakes):
                on = time >= burnout + cfg["deploy_delay_after_burnout_s"] and state[5] > 0
                air_brakes.deployment_level = fixed if on else 0.0
                log.append((time, state[2] - env.elevation, state[5],
                            air_brakes.deployment_level, np.nan))
                return time, air_brakes.deployment_level

        rocket.add_air_brakes(
            drag_coefficient_curve=lambda dep, mach: brake_cd(dep, mach),
            controller_function=ctrl,
            sampling_rate=cfg["controller_hz"],
            reference_area=flap_area(cfg),
            clamp=True,
            initial_observed_variables=[0, 0, 0],
            override_rocket_drag=False,
            name="Placeholder airbrakes",
        )
    fl_p = params["flight"]
    flight = Flight(rocket=rocket, environment=env, rail_length=fl_p["rail_length"],
                    inclination=min(fl_p["inclination"], 89.99), heading=fl_p["heading"],
                    terminate_on_apogee=True, max_time=60)
    return env, rocket, flight, np.array(log) if log else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("params", nargs="?", default="parameters.json")
    ap.add_argument("--target", type=float, help="target apogee, m AGL")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    cfg = dict(AIRBRAKE)
    if args.target:
        cfg["target_apogee_m"] = args.target
    params_path = Path(args.params).resolve()
    base_dir = params_path.parent
    params = json.loads(params_path.read_text())
    target = cfg["target_apogee_m"]

    print(f"Flap area {flap_area(cfg) * 1e4:.1f} cm²  "
          f"(body cross-section {np.pi * params['rocket']['radius'] ** 2 * 1e4:.1f} cm²)")

    env, rocket, base, _ = fly(params, base_dir, cfg, "none")
    _, _, ctl, log = fly(params, base_dir, cfg, "control")
    base_apo, ctl_apo = base.apogee - env.elevation, ctl.apogee - env.elevation

    levels = [0.25, 0.5, 0.75, 1.0]
    sweep = [(0.0, base_apo)]
    for lvl in levels:
        _, _, f, _ = fly(params, base_dir, cfg, lvl)
        sweep.append((lvl, f.apogee - env.elevation))
    sweep = np.array(sweep)

    print(f"\nNo brakes apogee:        {base_apo:7.1f} m")
    print(f"Target apogee:           {target:7.1f} m")
    print(f"With airbrakes apogee:   {ctl_apo:7.1f} m  (error {ctl_apo - target:+.1f} m)")
    print(f"Max deployment used:     {log[:, 3].max() * 100:5.0f} %")
    print("\nFixed-deployment sweep (brakes open from burnout to apogee):")
    for lvl, a in sweep:
        print(f"  {lvl * 100:5.0f} %  ->  {a:7.1f} m  ({a - base_apo:+6.1f} m)")

    # ------------------------------------------------------------------ plot
    blue, orange, gray, green, red = "#2563eb", "#ea580c", "#6b7280", "#16a34a", "#dc2626"
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axs = plt.subplots(2, 2, figsize=(14, 9.5), constrained_layout=True)
    fig.suptitle(f"Placeholder airbrakes — target {target:.0f} m, "
                 f"result {ctl_apo:.1f} m (no brakes {base_apo:.0f} m)",
                 fontsize=14, fontweight="bold")
    burnout = rocket.motor.burn_out_time

    ax = axs[0, 0]
    t, z = fn_array(base.z)
    ax.plot(t, z - env.elevation, color=gray, lw=2, label=f"No brakes ({base_apo:.0f} m)")
    t, z = fn_array(ctl.z)
    ax.plot(t, z - env.elevation, color=blue, lw=2, label=f"With airbrakes ({ctl_apo:.0f} m)")
    ax.axhline(target, color=red, ls="--", lw=1.5, label=f"Target {target:.0f} m")
    ax.axvline(burnout, color=gray, ls=":", lw=1)
    ax.set(title="Altitude (AGL)", xlabel="Time (s)", ylabel="Altitude (m)", ylim=(0, None))
    ax.legend(loc="lower right")

    ax = axs[0, 1]
    ax.plot(log[:, 0], log[:, 3] * 100, color=orange, lw=2, drawstyle="steps-post")
    ax.axvline(burnout, color=gray, ls=":", lw=1)
    ax.text(burnout, 95, " burnout", color=gray, va="top")
    ax.set(title=f"Airbrake deployment ({cfg['controller_hz']} Hz controller)",
           xlabel="Time (s)", ylabel="Deployment (%)", ylim=(-3, 103))

    ax = axs[1, 0]
    m = log[:, 0] >= burnout
    ax.plot(log[m, 0], log[m, 4], color=green, lw=2, label="Predicted apogee (in flight)")
    ax.axhline(target, color=red, ls="--", lw=1.5, label="Target")
    ax.axhline(ctl_apo, color=blue, ls=":", lw=1.5, label="Actual apogee")
    ax.set(title="Controller's apogee prediction", xlabel="Time (s)", ylabel="Altitude (m)")
    ax.legend(loc="upper right")

    ax = axs[1, 1]
    ax.plot(sweep[:, 0] * 100, sweep[:, 1], "o-", color=blue, lw=2)
    for lvl, a in sweep:
        ax.annotate(f"{a:.0f} m", (lvl * 100, a), textcoords="offset points",
                    xytext=(0, 8), ha="center")
    ax.axhline(target, color=red, ls="--", lw=1.5, label="Target")
    ax.set(title="Authority: apogee vs fixed deployment", xlabel="Deployment held after burnout (%)",
           ylabel="Apogee (m)")
    ax.legend(loc="upper right")
    ax.text(0.02, 0.04, f"{cfg['n_flaps']} flaps × {cfg['flap_width_m'] * 100:.1f} × "
            f"{cfg['flap_extension_m'] * 100:.1f} cm, Cd {cfg['cd_full']} (placeholder)",
            transform=ax.transAxes, color=gray)

    out = base_dir / "airbrakes_results.png"
    fig.savefig(out, dpi=150)
    print(f"\nSaved figure -> {out}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()