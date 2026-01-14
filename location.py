import pandas as pd
import random

# 50 różnorodnych wzorców lokalizacji
location_base_reviews = [
    # POZYTYWNE (Dostępność, centrum, widoki)
    ["Great location near metro station very central walk everywhere.", 5, 1],
    ["Perfect spot close to underground and bus stops easy transport.", 5, 1],
    ["Amazing view of the city from our window central position.", 5, 1],
    ["Heart of the city short distance to all tourist attractions.", 5, 1],
    ["Conveniently located near the tube station and local markets.", 4, 1],
    ["Beautiful view of the Eiffel Tower right from the balcony.", 5, 1],
    ["Excellent position right next to the museum and parks.", 5, 1],
    ["Right in the middle of the shopping district, so convenient.", 5, 1],
    ["Easy walk to the Louvre and other famous landmarks.", 4, 1],
    ["Spectacular view from the top floor room, simply stunning.", 5, 1],
    ["Minutes away from the nearest subway stop, very easy travel.", 5, 1],
    ["Great base for exploring the city on foot.", 4, 1],
    ["Prime location for sightseeing and great nightlife nearby.", 5, 1],
    ["Right by the river with stunning sunset views.", 5, 1],
    ["Located in a lovely quiet street but just steps from the action.", 4, 1],
    ["Superb transport links, the train station is just around the corner.", 5, 1],
    ["Walking distance to the convention center and main business hub.", 4, 1],
    ["Fantastic panoramic views of the entire skyline.", 5, 1],
    ["Ideal spot for foodies, surrounded by the best local restaurants.", 5, 1],
    ["The location is unbeatable, right across from the main square.", 5, 1],

    # NEGATYWNE (Dystans, izolacja, dojazd, otoczenie)
    ["Hotel is far from center and airport taxi was very expensive.", 2, 0],
    ["Impossible to find, difficult location far from any station.", 1, 0],
    ["Too far from the city center, spent too much on trains.", 2, 0],
    ["The view was just a brick wall and a dumpster, very disappointing.", 1, 0],
    ["Far from shops and restaurants, had to walk 20 minutes to eat.", 2, 0],
    ["Isolated location, absolutely nothing to do nearby.", 1, 0],
    ["Located in a shady neighborhood, felt very unsafe at night.", 1, 0],
    ["Hard to get to by public transport, needed a car for everything.", 2, 0],
    ["Miles away from anything interesting, terrible spot.", 1, 0],
    ["Nightmare to find the hotel in these tiny side streets.", 2, 0],
    ["Very far from the main attractions, do not stay here for tourism.", 1, 0],
    ["Inconvenient location for tourists, took hours to get anywhere.", 2, 0],
    ["Nothing but factories around, depressing industrial view.", 1, 0],
    ["Long hike uphill from the metro, not recommended with luggage.", 2, 0],
    ["The surrounding area felt sketchy and poorly lit at night.", 1, 0],
    ["Out in the suburbs, very expensive to travel to the center.", 2, 0],
    ["No windows in the room, so no view at all, felt like a basement.", 1, 0],
    ["The maps are wrong, the hotel is much further than advertised.", 2, 0],
    ["Trapped in a tourist trap area with overpriced everything.", 2, 0],
    ["Constant noise from the highway right outside the window.", 1, 0],

    # MIESZANE / SPECYFICZNE
    ["Noisy street but very close to the main square and shops.", 3, 1],
    ["Tucked away in a small side street, very central but quiet.", 4, 1],
    ["Located on a steep hill, difficult to walk up with bags.", 3, 0],
    ["Bit of a hike from the station, but the view was worth it.", 3, 1],
    ["Out of town location but good shuttle service provided.", 3, 1],
    ["Great view of the harbor, but the area is very crowded.", 3, 1],
    ["Central enough to walk, but far enough to be quiet at night.", 4, 1],
    ["The location is okay if you only need the airport nearby.", 3, 1],
    ["Good for a business trip, but too far for sightseeing.", 3, 0],
    ["Beautiful scenery around, but zero public transport links.", 3, 0]
]

# rozszerona lista (500 rekordów)
final_rows = []
category_name = "Location & View"

for _ in range(500):
    base = random.choice(location_base_reviews)
    
    variations = [
        " Definitely the best part of the hotel.",
        " Perfect for our weekend getaway.",
        " We loved being so close to everything.",
        " A bit of a trade-off between price and distance.",
        " The metro station made all the difference.",
        " We spent a fortune on Uber because it's so far.",
        " Just follow the maps carefully.",
        " Stunning at night!",
        " Not the most glamorous neighborhood.",
        " Everything is within a 5-minute walk.",
        " I would choose this spot again just for the view.",
        " Good for those who like to explore on foot."
    ]
    
    new_text = base[0] + random.choice(variations)
    final_rows.append([new_text, base[1], base[2], category_name])

# tworzenie dataframe i zapis do csv
df_location = pd.DataFrame(final_rows, columns=['Review', 'Rating', 'Sentiment', 'category'])
df_location.to_csv('location_augmentation.csv', index=False)
