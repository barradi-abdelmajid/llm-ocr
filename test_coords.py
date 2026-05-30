import fitz

doc = fitz.open()
page = doc.new_page(width=612, height=803)

# Draw a red rect at a known position (top-left area)
shape = page.new_shape()
shape.draw_rect(fitz.Rect(50, 700, 200, 780))
shape.finish(color=(1, 0, 0), width=2)
shape.commit()

# Insert text at the SAME position
page.insert_text(
    (50, 750),
    "TEST HERE",
    fontsize=12,
    fontname="helv",
    fill_color=(0, 0, 1),
)

# Draw a green rect at bottom-right
shape = page.new_shape()
shape.draw_rect(fitz.Rect(400, 20, 580, 100))
shape.finish(color=(0, 1, 0), width=2)
shape.commit()

page.insert_text(
    (400, 70),
    "TEST BOTTOM",
    fontsize=12,
    fontname="helv",
    fill_color=(0, 0, 1),
)

doc.save("phases/test_coords.pdf")
doc.close()
print("Saved phases/test_coords.pdf")
