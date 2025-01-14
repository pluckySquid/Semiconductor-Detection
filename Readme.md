This is a very simple readme.

# Get Legend
```
cd fitting_curve
python find_legend.py
```
This would create a a output.py which is the left 12% bot 6% of the original pdf, which is the entire legend. Which can be further used to auto detect semiconductor examples.

# auto_exact
```
python3 src/auto_extract.py
```
This will create a processed image in data/islands_with_straight_and_curved_lines.png, which the curved lines will be plot in blue and stright lines will be plot in green. Each identified semiconductors will be stored in data/auto_extract, where the unique and duplicated semiconductors will be stored seperately. 

# auto_identify
```
python src/auto_identify.py
```

THe current setting is using images in the data/manual folder to identify semicondutors, you can modified the code to use different templates to identify semicondutors.

# remove letters
src/remove_letters.py is used to remove letters and numbers

# extract_semicondictor_from_pdf
```
python fitting_curve/extract_semicondictor_from_pdf.py
```
Use this to extract semicondutors from the legend.