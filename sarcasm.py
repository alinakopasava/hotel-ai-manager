import pandas as pd
import random

# 22 różnorodnych wzorców sarcasmu 
sarcastic_reviews = [
    # CLEANLINESS
    ["The carpet was a fascinating ecosystem of stains from the mid-90s. Truly a vintage experience.", 0, "Cleanliness"],
    ["Loved the personal touch of finding the previous guest's hair on my 'fresh' pillow. So intimate.", 0, "Cleanliness"],
    ["The bathroom was a lovely shade of swamp green. I didn't know I booked a botanical garden stay.", 0, "Cleanliness"],
    ["If you enjoy the rustic scent of old cigarettes and damp mold, this place is a absolute paradise.", 0, "Cleanliness"],
    ["The housekeeping is so stealthy they haven't entered my room in four days. Amazing privacy!", 0, "Cleanliness"],
    
    # SERVICE & MANAGEMENT
    ["The receptionist was a master of the 'invisible guest' technique. I stood there for 20 minutes unnoticed.", 0, "Service & Management"],
    ["I've never felt more like a burden than when I asked for an extra towel. Outstanding hospitality.", 0, "Service & Management"],
    ["The manager's ability to ignore a direct question is truly world-class. A masterclass in silence.", 0, "Service & Management"],
    ["Waiting two hours for a cold sandwich was the highlight of my culinary journey here. Lightning fast!", 0, "Service & Management"],
    ["Staff was so professional they managed to lose my luggage and my will to live in the same hour.", 0, "Service & Management"],
    
    # PRICE & VALUE
    ["Fifty dollars for a bottle of water? I feel so privileged to contribute to the hotel's retirement fund.", 0, "Price & Value"],
    ["The 'luxury' surcharge for a room with no windows was a brilliant joke. I'm still laughing.", 0, "Price & Value"],
    ["I love paying five-star prices for a one-star experience. It's a very unique financial strategy.", 0, "Price & Value"],
    ["The hidden fees at checkout were like a fun treasure hunt, except the treasure was my empty wallet.", 0, "Price & Value"],
    
    # AMENITIES & ROOM
    ["The Wi-Fi is perfect if you want to experience the internet at the speed of the 19th century.", 0, "Amenities & Room"],
    ["The gym consisted of a single rusty dumbbell and a broken dream. Truly a fitness lover's dream.", 0, "Amenities & Room"],
    ["The air conditioning sounds like a jet engine taking off. Very soothing for sleep.", 0, "Amenities & Room"],
    
    # FOOD & BREAKFAST
    ["The scrambled eggs had the consistency of a rubber tire. High quality culinary art.", 0, "Food & Breakfast"],
    ["I enjoyed the mystery meat at the buffet. A real adventure for my stomach.", 0, "Food & Breakfast"],
    
    # LOCATION & VIEW
    ["The 'ocean view' was a masterpiece of imagination, mostly involving a brick wall and a dumpster.", 0, "Location & View"],
    ["Perfect location if you enjoy the soothing, rhythmic sounds of a 24-hour construction site.", 0, "Location & View"],
    ["The central location is great if you want to be in the middle of a high-crime area. Lively!", 0, "Location & View"]
]

# rozszerona lista (250 rekordów)
final_rows = []
for _ in range(250):
    base = random.choice(sarcastic_reviews)
    variations = [" Honestly.", " What a joke.", " Simply stunning.", " Just wow.", " Brilliant.", " Incredible."]
    new_text = base[0] + random.choice(variations)
    final_rows.append([new_text, base[1], base[2]])

# tworzenie dataframe
df_sarcasm = pd.DataFrame(final_rows, columns=['Review', 'Sentiment', 'category'])

# dodanie kolumny rating z wartością 1
df_sarcasm['Rating'] = 1

# zapis do pliku 
df_sarcasm.to_csv('sarcasm_augmentation.csv', index=False)