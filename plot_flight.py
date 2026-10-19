"""
Run a RocketPy flight from an ork2json parameters.json and plot the results
against the OpenRocket numbers stored in the same file.

Usage:
    python plot_flight.py                      # uses ./parameters.json
    python plot_flight.py subscale/parameters.json
    python plot_flight.py parameters.json --no-show   # just save the PNG/CSV

Outputs (next to parameters.json):
    flight_results.png   - 6-panel figure + RocketPy vs OpenRocket table
    flight_data.csv      - time, altitude, velocity, acceleration, ...
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rocketpy import Environment, Flight, Function, Rocket, SolidMotor

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def find_file(path_str, base_dir):
    """ork2json writes paths relative to where it was run; try a few places."""
    p = Path(path_str)
    candidates = [p, base_dir / p, base_dir / p.name, base_dir / "subscale" / p.name]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Could not find {path_str} (looked in {base_dir})")


def load_curve(path):
    """Load a 2-column CSV, sort it, and average duplicate x values."""
    data = np.loadtxt(path, delimiter=",")
    data = data[np.argsort(data[:, 0], kind="stable")]
    x_unique, inv = np.unique(data[:, 0], return_inverse=True)
    y_mean = np.bincount(inv, weights=data[:, 1]) / np.bincount(inv)
    return data, np.column_stack([x_unique, y_mean])


def fn_array(func):
    """RocketPy Function -> (t, y) numpy arrays."""
    src = np.asarray(func.source)
    return src[:, 0], src[:, 1]


# ----------------------------------------------------------------------------
# Build and fly
# ----------------------------------------------------------------------------


def build_and_fly(params, base_dir):
    env_p, rkt_p, mot_p = params["environment"], params["rocket"], params["motors"]

    # --- Environment: standard atmosphere + constant wind (OpenRocket default style)
    env = Environment(
        latitude=env_p["latitude"],
        longitude=env_p["longitude"],
        elevation=env_p["elevation"] or 0.0,
    )
    env.set_atmospheric_model(
        type="custom_atmosphere",
        wind_u=env_p.get("wind_average") or 0.0,  # m/s, blowing toward +x (east)
        wind_v=0.0,
    )

    # --- Motor
    thrust_path = find_file(mot_p["thrust_source"], base_dir)
    thrust_raw, _ = load_curve(thrust_path)
    motor = SolidMotor(
        thrust_source=str(thrust_path),
        dry_mass=mot_p["dry_mass"],
        dry_inertia=tuple(mot_p["dry_inertia"]),
        nozzle_radius=mot_p["nozzle_radius"],
        throat_radius=mot_p["throat_radius"],
        grain_number=mot_p["grain_number"],
        grain_density=mot_p["grain_density"],
        grain_outer_radius=mot_p["grain_outer_radius"],
        grain_initial_inner_radius=mot_p["grain_initial_inner_radius"],
        grain_initial_height=mot_p["grain_initial_height"],
        grain_separation=mot_p["grain_separation"],
        grains_center_of_mass_position=mot_p["grains_center_of_mass_position"],
        center_of_dry_mass_position=mot_p["center_of_dry_mass_position"],
        nozzle_position=mot_p["nozzle_position"],
        coordinate_system_orientation=mot_p["coordinate_system_orientation"],
    )

    # --- Drag: the OpenRocket export has duplicate Mach values -> clean it
    drag_raw, drag_clean = load_curve(find_file(rkt_p["drag_curve"], base_dir))

    # --- Inertia: RocketPy wants (I11, I22, I33) with I33 about the long axis.
    # ork2json sometimes writes the large (pitch/yaw) value last -> fix it.
    inertia = list(rkt_p["inertia"])
    if inertia[2] > inertia[0]:
        print(f"[note] Inertia looked swapped {inertia}; using "
              f"{[inertia[2], inertia[2], inertia[0]]}")
        inertia = [inertia[2], inertia[2], inertia[0]]

    rocket = Rocket(
        radius=rkt_p["radius"],
        mass=rkt_p["mass"],
        inertia=tuple(inertia),
        power_off_drag=Function(drag_clean, "Mach", "Cd", extrapolation="constant"),
        power_on_drag=Function(drag_clean, "Mach", "Cd", extrapolation="constant"),
        center_of_mass_without_motor=rkt_p["center_of_mass_without_propellant"],
        coordinate_system_orientation=rkt_p["coordinate_system_orientation"],
    )
    rocket.add_motor(motor, position=mot_p["position"])

    nose = params["nosecones"]
    # OpenRocket and RocketPy name a couple of shapes differently
    kind_map = {"ellipsoid": "elliptical", "haack": "vonkarman", "power": "powerseries"}
    kind = kind_map.get(nose["kind"].lower(), nose["kind"].lower())
    rocket.add_nose(length=nose["length"], kind=kind, position=nose["position"],
                    name=nose.get("name", "Nose Cone"))

    for fin in params.get("trapezoidal_fins", {}).values():
        rocket.add_trapezoidal_fins(
            n=fin["number"], root_chord=fin["root_chord"], tip_chord=fin["tip_chord"],
            span=fin["span"], position=fin["position"], cant_angle=fin["cant_angle"],
            sweep_length=fin["sweep_length"], name=fin.get("name", "Fins"),
        )

    for tail in params.get("tails", {}).values():
        rocket.add_tail(top_radius=tail["top_radius"], bottom_radius=tail["bottom_radius"],
                        length=tail["length"], position=tail["position"])

    rb = params.get("rail_buttons")
    if rb:
        rocket.set_rail_buttons(upper_button_position=rb["upper_position"],
                                lower_button_position=rb["lower_position"],
                                angular_position=rb.get("angular_position", 45) or 45)

    for chute in params.get("parachutes", {}).values():
        trigger = "apogee" if chute["deploy_event"] == "apogee" else float(chute["deploy_altitude"])
        rocket.add_parachute(name=chute["name"].split("[")[0].strip() + f" ({trigger})",
                             cd_s=chute["cds"], trigger=trigger, sampling_rate=105,
                             lag=chute.get("deploy_delay") or 0.0)

    # --- Flight (exactly 90 deg can be numerically awkward; 89.99 is equivalent)
    fl_p = params["flight"]
    flight = Flight(
        rocket=rocket, environment=env, rail_length=fl_p["rail_length"],
        inclination=min(fl_p["inclination"], 89.99), heading=fl_p["heading"],
        max_time=600,
    )
    return env, rocket, motor, flight, thrust_raw, drag_raw, drag_clean


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------


def make_figure(params, env, rocket, flight, thrust_raw, drag_raw, drag_clean, out_png):
    orr = params.get("stored_results", {})
    elev = env.elevation

    t, z = fn_array(flight.z)
    alt = z - elev
    tv, vz = fn_array(flight.vz)
    ts, speed = fn_array(flight.speed)
    ta, acc = fn_array(flight.acceleration)
    t_apo, apo = flight.apogee_time, flight.apogee - elev

    blue, orange, gray, green = "#2563eb", "#ea580c", "#6b7280", "#16a34a"
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
                         "axes.spines.top": False, "axes.spines.right": False})

    fig = plt.figure(figsize=(16, 11), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.55])
    name = params.get("id", {}).get("rocket_name", "Rocket")
    fig.suptitle(f"{name} — RocketPy flight vs OpenRocket", fontsize=15, fontweight="bold")

    # 1. Altitude
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(t, alt, color=blue, lw=2, label="RocketPy")
    ax.plot(t_apo, apo, "o", color=blue)
    ax.annotate(f"Apogee {apo:.0f} m\n@ {t_apo:.1f} s", (t_apo, apo),
                textcoords="offset points", xytext=(10, -30), color=blue)
    if "max_altitude" in orr:
        ax.axhline(orr["max_altitude"], color=orange, ls="--", lw=1.5,
                   label=f"OpenRocket apogee {orr['max_altitude']:.0f} m")
    ax.set(title="Altitude (AGL)", xlabel="Time (s)", ylabel="Altitude (m)", ylim=(0, None))
    ax.legend(loc="upper right")

    # 2. Ascent zoom: altitude to apogee
    ax = fig.add_subplot(gs[0, 1])
    m = t <= t_apo * 1.15
    ax.plot(t[m], alt[m], color=blue, lw=2, label="RocketPy")
    if "max_altitude" in orr:
        ax.plot(orr.get("time_to_apogee", t_apo), orr["max_altitude"], "s",
                color=orange, ms=8, label="OpenRocket apogee")
    ax.plot(t_apo, apo, "o", color=blue, ms=8, label="RocketPy apogee")
    ax.axvline(flight.rocket.motor.burn_out_time, color=gray, ls=":", lw=1)
    ax.text(flight.rocket.motor.burn_out_time, 0.05 * apo, " burnout", color=gray)
    ax.set(title="Ascent to apogee", xlabel="Time (s)", ylabel="Altitude (m)", ylim=(0, None))
    ax.legend(loc="lower right")

    # 3. Velocity
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(ts, speed, color=blue, lw=2, label="Total speed")
    ax.plot(tv, vz, color=green, lw=1.5, alpha=0.8, label="Vertical velocity")
    if "max_velocity" in orr:
        ax.axhline(orr["max_velocity"], color=orange, ls="--", lw=1.5,
                   label=f"OpenRocket max {orr['max_velocity']:.0f} m/s")
    ax.axhline(0, color="k", lw=0.6)
    ax.set(title="Velocity", xlabel="Time (s)", ylabel="m/s")
    ax.legend(loc="upper right")

    # 4. Acceleration (first ~ apogee time)
    ax = fig.add_subplot(gs[1, 0])
    m = ta <= t_apo
    ax.plot(ta[m], acc[m], color=blue, lw=2, label="RocketPy")
    if "max_acceleration" in orr:
        ax.axhline(orr["max_acceleration"], color=orange, ls="--", lw=1.5,
                   label=f"OpenRocket max {orr['max_acceleration']:.0f} m/s²")
    ax.set(title="Acceleration (to apogee)", xlabel="Time (s)", ylabel="m/s²")
    ax.legend(loc="upper right")

    # 5. Thrust curve
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(thrust_raw[:, 0], thrust_raw[:, 1], color=orange, lw=2)
    ax.fill_between(thrust_raw[:, 0], thrust_raw[:, 1], color=orange, alpha=0.15)
    impulse = np.trapezoid(thrust_raw[:, 1], thrust_raw[:, 0]) if hasattr(np, "trapezoid") \
        else np.trapz(thrust_raw[:, 1], thrust_raw[:, 0])
    ax.text(0.97, 0.95, f"Total impulse {impulse:.0f} N·s\nPeak {thrust_raw[:, 1].max():.0f} N"
            f"\nBurn {thrust_raw[-1, 0]:.2f} s", transform=ax.transAxes, ha="right", va="top")
    ax.set(title="Motor thrust curve", xlabel="Time (s)", ylabel="Thrust (N)", ylim=(0, None))

    # 6. Drag curve
    ax = fig.add_subplot(gs[1, 2])
    ax.scatter(drag_raw[:, 0], drag_raw[:, 1], s=6, color=gray, alpha=0.5, label="OpenRocket export")
    ax.plot(drag_clean[:, 0], drag_clean[:, 1], color=blue, lw=1.5, label="Used in RocketPy")
    ax.set(title="Drag coefficient vs Mach", xlabel="Mach", ylabel="Cd", ylim=(0.4, 1.05))
    ax.legend(loc="upper right")

    # 7. Comparison table
    ax = fig.add_subplot(gs[2, :])
    ax.axis("off")
    rows = [
        ("Apogee (m)", apo, orr.get("max_altitude")),
        ("Time to apogee (s)", t_apo, orr.get("time_to_apogee")),
        ("Max velocity (m/s)", flight.max_speed, orr.get("max_velocity")),
        ("Max Mach", flight.max_mach_number, orr.get("max_mach")),
        ("Max acceleration (m/s²)", flight.max_acceleration, orr.get("max_acceleration")),
        ("Rail exit velocity (m/s)", flight.out_of_rail_velocity, orr.get("launch_rod_velocity")),
        ("Flight time (s)", flight.t_final, orr.get("flight_time")),
        ("Ground hit velocity (m/s)", abs(flight.impact_velocity), orr.get("ground_hit_velocity")),
    ]
    cell = []
    for label, rp, orv in rows:
        diff = f"{(rp - orv) / orv * 100:+.1f}%" if orv else "—"
        cell.append([label, f"{rp:.2f}", f"{orv:.2f}" if orv is not None else "—", diff])
    tbl = ax.table(cellText=cell, colLabels=["Metric", "RocketPy", "OpenRocket", "Difference"],
                   loc="center", cellLoc="center", colWidths=[0.28, 0.16, 0.16, 0.14])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10.5)
    tbl.scale(1, 1.45)
    for (r, c), cobj in tbl.get_celld().items():
        cobj.set_edgecolor("#d1d5db")
        if r == 0:
            cobj.set_facecolor("#1f2937")
            cobj.set_text_props(color="white", fontweight="bold")
        elif r == 1:
            cobj.set_facecolor("#eff6ff")
            cobj.set_text_props(fontweight="bold")

    fig.savefig(out_png, dpi=150)
    print(f"Saved figure -> {out_png}")
    return fig, rows


def save_csv(flight, env, out_csv):
    t, z = fn_array(flight.z)
    cols = {
        "altitude_agl_m": z - env.elevation,
        "vz_m_s": np.interp(t, *fn_array(flight.vz)),
        "speed_m_s": np.interp(t, *fn_array(flight.speed)),
        "acceleration_m_s2": np.interp(t, *fn_array(flight.acceleration)),
        "mach": np.interp(t, *fn_array(flight.mach_number)),
        "x_m": np.interp(t, *fn_array(flight.x)),
        "y_m": np.interp(t, *fn_array(flight.y)),
    }
    data = np.column_stack([t] + list(cols.values()))
    np.savetxt(out_csv, data, delimiter=",", header="time_s," + ",".join(cols), comments="",
               fmt="%.5f")
    print(f"Saved data   -> {out_csv}")


# ----------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("params", nargs="?", default="parameters.json")
    ap.add_argument("--no-show", action="store_true", help="don't open the plot window")
    args = ap.parse_args()

    params_path = Path(args.params).resolve()
    base_dir = params_path.parent
    params = json.loads(params_path.read_text())

    env, rocket, motor, flight, thrust_raw, drag_raw, drag_clean = build_and_fly(params, base_dir)

    print(f"\nStatic margin at liftoff: {rocket.static_margin(0):.2f} cal")
    fig, rows = make_figure(params, env, rocket, flight, thrust_raw, drag_raw, drag_clean,
                            base_dir / "flight_results.png")
    save_csv(flight, env, base_dir / "flight_data.csv")

    print(f"\n{'Metric':28s}{'RocketPy':>12s}{'OpenRocket':>12s}{'Diff':>9s}")
    for label, rp, orv in rows:
        d = f"{(rp - orv) / orv * 100:+.1f}%" if orv else "—"
        print(f"{label:28s}{rp:12.2f}{(orv if orv is not None else float('nan')):12.2f}{d:>9s}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()