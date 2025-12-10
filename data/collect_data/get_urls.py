from pyinaturalist import get_observations
import pandas as pd
import time
import random
import os

# CONFIGURATION
TARGET_IMAGES_PER_CLASS = 2000  # Cap to balance dataset
PLACE_ID_NC = 30
PLACE_ID_USA = 1

# Define which classes need USA expansion
EXPAND_TO_USA = [8, 9, 10] # Blister Beetles, Scorpions, Horse Flies

# IDs found via iNaturalist taxonomy search
CLASSES = {
    0:  {'name': 'Venomous Spiders',   'taxon_ids': [47370, 48140]}, # Latrodectus (not Theridiidae as this includes non harmful), Sicariidae
    1:  {'name': 'Ticks',              'taxon_ids': [51673]},        # Ixodidae 
    2:  {'name': 'Mosquitoes',         'taxon_ids': [52134]},        # Culicidae
    3:  {'name': 'Stinging Wasps',     'taxon_ids': [52747]},        # Vespidae
    4:  {'name': 'Bees',               'taxon_ids': [47221]},        # Apidae
    5:  {'name': 'Fire Ants',          'taxon_ids': [67597]},        # Genus Solenopsis
    6:  {'name': 'Assassin Bugs',      'taxon_ids': [48959]},        # Reduviidae
    7:  {'name': 'Venomous Cats.',     'taxon_ids': [84186, 84165]}, # Megalopygidae, Limacodidae 
    8:  {'name': 'Blister Beetles',    'taxon_ids': [59510]},        # Meloidae 
    9:  {'name': 'Scorpions',          'taxon_ids': [52572]},        # Vaejovidae 
    10: {'name': 'Horse/Deer Flies',   'taxon_ids': [47821]},        # Tabanidae 
    11: {'name': 'Centipedes',         'taxon_ids': [49556]}         # Order Chilopoda
}

dataset_rows = []

print(f"Starting collection... Target: {TARGET_IMAGES_PER_CLASS} per class.\n")

for class_id, info in CLASSES.items():
    
    # 1. Determine Geography
    place_id = PLACE_ID_USA if class_id in EXPAND_TO_USA else PLACE_ID_NC
    location_name = "USA" if class_id in EXPAND_TO_USA else "NC"
    
    print(f"Collecting Class {class_id} ({info['name']}) from {location_name}...")
    
    collected_urls = set()
    page = 1
    
    # Loop through taxon IDs for this class (e.g. Spiders has 2 IDs)
    for taxon_id in info['taxon_ids']:
        
        while len(collected_urls) < TARGET_IMAGES_PER_CLASS:
            try:
                observations = get_observations(
                    taxon_id=taxon_id,
                    place_id=place_id,
                    quality_grade="research",
                    term_id=1 if class_id == 7 else None, # Class 7 Larva filter
                    term_value_id=6 if class_id == 7 else None,
                    page=page,
                    per_page=200 # Max allowed per page
                )
                
                obs_list = observations['results']
                
                if not obs_list:
                    break # No more data available
                
                for obs in obs_list:
                    if len(collected_urls) >= TARGET_IMAGES_PER_CLASS:
                        break
                    
                    # EXTRACT IMAGE URL
                    # We grab the 'medium' size image (approx 500x500)
                    if obs['photos']:
                        photo_url = obs['photos'][0]['url']
                        # By default iNat returns 'square' (75x75). Change to 'medium'.
                        medium_url = photo_url.replace("square", "medium")
                        
                        # Store metadata
                        dataset_rows.append({
                            'class_id': class_id,
                            'class_name': info['name'],
                            'taxon_id': taxon_id,
                            'obs_id': obs['id'],
                            'image_url': medium_url,
                            'source_loc': location_name
                        })
                        collected_urls.add(medium_url)
                
                print(f"  -> Page {page}: Total collected so far: {len(collected_urls)}")
                page += 1
                time.sleep(2.3 + random.random()) # Be polite to API
                
            except Exception as e:
                print(f"  [!] Error on page {page}: {e}")
                break
                
    print(f"  [OK] Finished Class {class_id}. Total: {len(collected_urls)}\n")

# Save to CSV
df = pd.DataFrame(dataset_rows)
df.to_csv("../nc_arthropod_dataset_urls.csv", index=False)
print("Done! URLs saved to 'nc_arthropod_dataset_urls.csv'")