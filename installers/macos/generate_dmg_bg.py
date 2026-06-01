#!/usr/bin/env python3
"""
SUDARSHAN AI - DMG Background Builder
==============================================================================
Generates a stunning, modern dark glassmorphic background for the DMG installer
volume using Pillow. Features ambient cyan/purple neon glows, grid layouts,
drop-zone rounded squares, and clear drag-and-drop instructions.
==============================================================================
"""

import os
from PIL import Image, ImageDraw, ImageFont

def draw_rounded_rectangle(draw, xy, corner_radius, fill=None, outline=None, width=1):
    """Draw a rounded rectangle using PIL."""
    draw.rounded_rectangle(xy, radius=corner_radius, fill=fill, outline=outline, width=width)

def generate_background():
    # DMG standard resolution
    width = 600
    height = 400
    
    # Create master image
    img = Image.new("RGBA", (width, height), (15, 23, 42, 255)) # Dark slate #0F172A
    draw = ImageDraw.Draw(img)
    
    # 1. Create futuristic circular neon glows in background
    # We draw soft radial gradient circles
    glow_canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow_canvas)
    
    # Left Cyan Glow
    for r in range(120, 0, -2):
        alpha = int((1 - (r / 120)) * 25) # soft transparency
        glow_draw.ellipse([150 - r, 180 - r, 150 + r, 180 + r], fill=(6, 182, 212, alpha))
        
    # Right Purple Glow
    for r in range(120, 0, -2):
        alpha = int((1 - (r / 120)) * 20)
        glow_draw.ellipse([450 - r, 180 - r, 450 + r, 180 + r], fill=(139, 92, 246, alpha))
        
    # Merge glows onto main image
    img = Image.alpha_composite(img, glow_canvas)
    draw = ImageDraw.Draw(img)
    
    # 2. Draw modern grid lines
    grid_color = (30, 41, 59, 255) # Slate #1E293B
    for x in range(0, width, 40):
        draw.line([(x, 0), (x, height)], fill=grid_color, width=1)
    for y in range(0, height, 40):
        draw.line([(0, y), (width, y)], fill=grid_color, width=1)
        
    # 3. Load Fonts
    font_title = None
    font_sub = None
    font_labels = None
    
    # Standard macOS font paths
    font_paths = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttf"
    ]
    
    for path in font_paths:
        if os.path.exists(path):
            try:
                font_title = ImageFont.truetype(path, 22)
                font_sub = ImageFont.truetype(path, 13)
                font_labels = ImageFont.truetype(path, 14)
                break
            except:
                pass
                
    if not font_title:
        font_title = ImageFont.load_default()
        font_sub = ImageFont.load_default()
        font_labels = ImageFont.load_default()
        
    # 4. Draw Header Titles & Instructions
    draw.text((300, 35), "SUDARSHAN AI INSTALLER", fill=(248, 250, 252, 255), font=font_title, anchor="mm") # White #F8FAFC
    draw.text((300, 65), "Drag the application into the Applications folder to install it.", fill=(148, 163, 184, 255), font=font_sub, anchor="mm") # Slate #94A3B8
    
    # 5. Draw Glassmorphic Drop Zones
    # Left Box (App drop zone)
    draw_rounded_rectangle(draw, [60, 100, 240, 260], 12, fill=(30, 41, 59, 100), outline=(6, 182, 212, 120), width=2)
    # Right Box (Applications drop zone)
    draw_rounded_rectangle(draw, [360, 100, 540, 260], 12, fill=(30, 41, 59, 100), outline=(139, 92, 246, 100), width=2)
    
    # 6. Draw glowing arrows in center (Drag action guide)
    # Arrow line
    arrow_color = (6, 182, 212, 255) # Cyan
    draw.line([(260, 180), (340, 180)], fill=arrow_color, width=3)
    # Arrowhead
    draw.polygon([(340, 180), (330, 172), (330, 188)], fill=arrow_color)
    
    # Drag instruction text
    draw.text((300, 155), "drag to install", fill=(6, 182, 212, 200), font=font_sub, anchor="mm")
    
    # 7. Draw Labels below Drop Zones
    draw.text((150, 290), "SUDARSHAN AI.app", fill=(226, 232, 240, 255), font=font_labels, anchor="mm")
    draw.text((450, 290), "Applications", fill=(226, 232, 240, 255), font=font_labels, anchor="mm")
    
    # 8. Save output
    os.makedirs("../../assets", exist_ok=True)
    img.save("../../assets/dmg_background.png")
    print("[SUCCESS] DMG background image successfully written to assets/dmg_background.png!")

if __name__ == "__main__":
    generate_background()
