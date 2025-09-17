#!/usr/bin/env python3
"""
Create Pey Yar Logo as PNG
"""

try:
    from PIL import Image, ImageDraw, ImageFont
    import os
    
    def create_pey_yar_logo():
        # Create a 120x120 image with transparent background
        width, height = 120, 120
        image = Image.new('RGBA', (width, height), (255, 255, 255, 0))  # Transparent background
        draw = ImageDraw.Draw(image)
        
        # No background circle - completely transparent
        
        # Left Eye (P) - Blue
        draw.ellipse([33, 38, 57, 62], fill=(74, 144, 226), outline=(46, 91, 186), width=2)
        draw.ellipse([39, 44, 51, 56], fill=(46, 91, 186))
        
        # Right Eye (Q) - Green  
        draw.ellipse([63, 38, 87, 62], fill=(126, 211, 33), outline=(91, 165, 23), width=2)
        # Q tail
        draw.polygon([(75, 44), (78, 47), (75, 50), (72, 47)], fill=(91, 165, 23))
        
        # Forehead/Sun - Orange
        draw.ellipse([35, 30, 85, 50], fill=(255, 149, 0))
        
        # Smile
        draw.arc([40, 60, 80, 70], 0, 180, fill=(255, 149, 0), width=3)
        
        # Left Exclamation Mark
        draw.rectangle([25, 20, 29, 40], fill=(74, 144, 226))
        draw.ellipse([24, 42, 30, 48], fill=(135, 206, 235))
        
        # Right Exclamation Mark
        draw.rectangle([91, 20, 95, 40], fill=(126, 211, 33))
        draw.ellipse([90, 42, 96, 48], fill=(91, 165, 23))
        
        # Plant (Growth)
        draw.rectangle([35, 30, 43, 34], fill=(255, 149, 0))
        # Plant leaves
        draw.polygon([(39, 30), (39, 20), (36, 23), (42, 23)], fill=(126, 211, 33))
        draw.ellipse([36, 16, 40, 20], fill=(126, 211, 33))
        draw.ellipse([38, 16, 42, 20], fill=(126, 211, 33))
        draw.ellipse([37, 13, 41, 17], fill=(126, 211, 33))
        
        # Quantum Signals
        draw.line([85, 35, 95, 25], fill=(255, 149, 0), width=2)
        draw.ellipse([93, 23, 97, 27], fill=(255, 215, 0))
        draw.ellipse([88, 28, 92, 32], fill=(255, 215, 0))
        
        # Quantum Waves
        draw.arc([85, 37, 95, 43], 0, 180, fill=(255, 215, 0), width=2)
        draw.arc([85, 42, 95, 48], 0, 180, fill=(255, 215, 0), width=2)
        draw.arc([85, 47, 95, 53], 0, 180, fill=(255, 215, 0), width=2)
        
        # Persian Text
        try:
            # Try to use a font that supports Persian
            font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()
        
        # Draw "پی یار" text
        draw.text((60, 90), "پی یار", font=font, fill=(74, 144, 226), anchor="mm")
        draw.text((60, 105), "Pey Yar", font=font, fill=(126, 211, 33), anchor="mm")
        
        return image
    
    if __name__ == "__main__":
        logo = create_pey_yar_logo()
        logo.save("pey_yar_logo.png", "PNG")
        print("✅ Logo created: pey_yar_logo.png")
        
except ImportError:
    print("⚠️ PIL not available, creating simple text logo instead")
    
    # Fallback: Create a simple text-based logo
    with open("pey_yar_logo.txt", "w", encoding="utf-8") as f:
        f.write("""
╔══════════════════════════════════╗
║            پی یار                ║
║           Pey Yar                ║
║                                 ║
║    🎨 PQN.AI Logo Design        ║
║    💙 Blue + 💚 Green + 🧡 Orange║
║                                 ║
║    چهره دوستانه و هوشمند        ║
║    یار و همراه پای              ║
╚══════════════════════════════════╝
        """)
    print("✅ Simple logo created: pey_yar_logo.txt")
