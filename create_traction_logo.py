#!/usr/bin/env python3
import svgwrite


def create_traction_logo(filename="assets/traction-logo.svg"):
    bg = "#0D2B2A"
    white = "#F3F7FA"
    green_dark = "#2D9B49"
    green_light = "#88D64A"
    amber = "#F2A90B"
    line_green = "#4E8D4A"

    width = 1400
    height = 420
    dwg = svgwrite.Drawing(filename, size=(width, height), profile="full")

    defs = dwg.defs
    grad_green = dwg.linearGradient(start=("0%", "0%"), end=("0%", "100%"), id="leafGreen")
    grad_green.add_stop_color("0%", green_light)
    grad_green.add_stop_color("100%", green_dark)
    defs.add(grad_green)

    grad_amber = dwg.linearGradient(start=("0%", "0%"), end=("0%", "100%"), id="leafAmber")
    grad_amber.add_stop_color("0%", "#FFC93A")
    grad_amber.add_stop_color("100%", amber)
    defs.add(grad_amber)

    dwg.add(defs)

    group = dwg.g(transform="translate(70,35)")
    group.add(dwg.circle(center=(155, 105), r=98, fill="none", stroke=line_green, stroke_width=3))

    group.add(dwg.path(
        d="M155 8 C112 44 95 71 95 110 C95 147 120 168 155 178 C190 168 215 147 215 110 C215 71 198 44 155 8 Z",
        fill="url(#leafGreen)",
    ))
    group.add(dwg.path(d="M155 109 C145 133 143 172 144 218 L166 218 C166 172 164 133 155 109 Z", fill=bg))
    group.add(dwg.path(d="M155 128 C147 114 133 99 112 88", stroke=bg, stroke_width=4, fill="none", stroke_linecap="round"))
    group.add(dwg.path(d="M155 128 C163 114 177 99 198 88", stroke=bg, stroke_width=4, fill="none", stroke_linecap="round"))

    group.add(dwg.path(
        d="M141 145 C91 133 53 139 34 170 C21 191 25 221 41 247 C65 286 105 297 143 290 C151 248 152 187 141 145 Z",
        fill="url(#leafGreen)",
        opacity=0.7,
    ))
    group.add(dwg.path(
        d="M170 145 C220 133 258 139 277 170 C290 191 286 221 270 247 C246 286 206 297 168 290 C160 248 159 187 170 145 Z",
        fill="url(#leafAmber)",
    ))

    group.add(dwg.line(start=(0, 285), end=(120, 285), stroke=line_green, stroke_width=4))
    group.add(dwg.line(start=(190, 285), end=(310, 285), stroke=line_green, stroke_width=4))
    group.add(dwg.path(
        d="M0 335 H110 Q120 335 128 340 H182 Q190 335 200 335 H310",
        stroke=line_green,
        stroke_width=4,
        fill="none",
    ))
    group.add(dwg.circle(center=(5, 335), r=8, fill=amber))
    group.add(dwg.circle(center=(305, 335), r=8, fill=amber))

    pin_group = dwg.g(transform="translate(155,337)")
    pin_group.add(dwg.circle(center=(0, 0), r=20, fill="#222930", stroke=white, stroke_width=4))
    pin_group.add(dwg.path(d="M-6 18 L0 30 L6 18 Z", fill=white))
    pin_group.add(dwg.circle(center=(8, 0), r=3.5, fill=amber))
    group.add(pin_group)
    dwg.add(group)

    dwg.add(dwg.text(
        "TRACTION",
        insert=(430, 245),
        fill=white,
        font_family="Helvetica Neue, Helvetica, Arial, sans-serif",
        font_size=170,
        font_weight=800,
        letter_spacing=3,
    ))

    dwg.save()
    print(f"Successfully generated logo: {filename}")


if __name__ == "__main__":
    create_traction_logo()
