import pandas as pd
import numpy as np

# Generate sample car data for testing
np.random.seed(42)

brands = ['Toyota', 'Mazda', 'Honda', 'Hyundai', 'Kia', 'Nissan', 'Suzuki',
          'Mitsubishi', 'Volkswagen', 'Skoda', 'Seat', 'Renault', 'Peugeot', 'Citroen',
          'BMW', 'Mercedes', 'Audi', 'Volvo', 'Ford', 'Chevrolet', 'Subaru', 'Lexus',
          'Opel', 'Fiat', 'Jeep', 'Mini', 'Porsche', 'Tesla', 'Land Rover', 'Jaguar']

models_map = {
    'Toyota': ['Corolla', 'Camry', 'RAV4', 'Yaris', 'Auris', 'C-HR', 'Prius', 'Highlander', 'Land Cruiser'],
    'Mazda': ['3', '6', 'CX-3', 'CX-5', 'CX-30', 'CX-9', 'MX-5', '323F'],
    'Honda': ['Civic', 'Accord', 'CR-V', 'Jazz', 'HR-V', 'Pilot', 'Odyssey'],
    'Hyundai': ['i20', 'i30', 'Tucson', 'Kona', 'Santa Fe', 'Elantra', 'Sonata', 'Palisade'],
    'Kia': ['Picanto', 'Rio', 'Ceed', 'Sportage', 'Sorento', 'Stinger', 'Seltos', 'Carnival'],
    'Nissan': ['Micra', 'Juke', 'Qashqai', 'X-Trail', 'Note', 'Leaf', 'Pathfinder'],
    'Suzuki': ['Swift', 'Vitara', 'S-Cross', 'Ignis', 'Baleno', 'Jimny'],
    'Mitsubishi': ['ASX', 'Outlander', 'Eclipse Cross', 'Space Star', 'Pajero'],
    'Volkswagen': ['Polo', 'Golf', 'Tiguan', 'Passat', 'T-Roc', 'Arteon', 'Touareg', 'ID.3', 'ID.4'],
    'Skoda': ['Fabia', 'Octavia', 'Superb', 'Karoq', 'Kodiaq', 'Kamiq', 'Enyaq'],
    'Seat': ['Ibiza', 'Leon', 'Arona', 'Ateca', 'Tarraco'],
    'Renault': ['Clio', 'Megane', 'Captur', 'Kadjar', 'Scenic', 'Zoe', 'Talisman'],
    'Peugeot': ['208', '308', '2008', '3008', '5008', '508', 'Rifter'],
    'Citroen': ['C3', 'C4', 'C5 Aircross', 'C3 Aircross', 'Berlingo'],
    'BMW': ['1 Series', '3 Series', '5 Series', 'X1', 'X3', 'X5', 'i3', 'iX3'],
    'Mercedes': ['A-Class', 'C-Class', 'E-Class', 'GLA', 'GLC', 'GLE', 'EQC'],
    'Audi': ['A3', 'A4', 'A6', 'Q3', 'Q5', 'Q7', 'e-tron'],
    'Volvo': ['V40', 'V60', 'S60', 'XC40', 'XC60', 'XC90'],
    'Ford': ['Fiesta', 'Focus', 'Puma', 'Kuga', 'Edge', 'Mustang'],
    'Chevrolet': ['Spark', 'Cruze', 'Malibu', 'Equinox', 'Traverse'],
    'Subaru': ['Impreza', 'XV', 'Forester', 'Outback', 'Legacy'],
    'Lexus': ['CT', 'IS', 'ES', 'NX', 'RX', 'UX'],
    'Opel': ['Corsa', 'Astra', 'Insignia', 'Mokka', 'Crossland', 'Grandland'],
    'Fiat': ['500', 'Panda', 'Tipo', '500X', 'Ducato'],
    'Jeep': ['Renegade', 'Compass', 'Cherokee', 'Grand Cherokee', 'Wrangler'],
    'Mini': ['Cooper', 'Countryman', 'Clubman'],
    'Porsche': ['911', 'Cayenne', 'Macan', 'Panamera', 'Taycan'],
    'Tesla': ['Model 3', 'Model Y', 'Model S', 'Model X'],
    'Land Rover': ['Range Rover', 'Range Rover Sport', 'Discovery', 'Defender', 'Evoque'],
    'Jaguar': ['XE', 'XF', 'F-Pace', 'E-Pace', 'I-Pace']
}

# Model rarity - affects price (1.0 = common, 1.5 = rare/premium)
model_rarity = {
    'Corolla': 1.0, 'Yaris': 1.0, 'Civic': 1.0, 'Golf': 1.0, 'Polo': 1.0,
    'Camry': 1.12, 'RAV4': 1.18, 'CR-V': 1.18, 'Tucson': 1.12, 'Qashqai': 1.08,
    'CX-5': 1.25, 'CX-30': 1.2, '3': 1.05, '6': 1.15, '323F': 0.85,  # Mazda models
    'Land Cruiser': 1.5, 'Prius': 1.25, 'Highlander': 1.35, 'Pajero': 1.3,
    '3 Series': 1.4, '5 Series': 1.5, 'X3': 1.45, 'X5': 1.55, 'X1': 1.35,
    'C-Class': 1.4, 'E-Class': 1.5, 'GLC': 1.45, 'GLE': 1.55, 'A-Class': 1.3,
    'A4': 1.4, 'A6': 1.5, 'Q5': 1.45, 'Q7': 1.55, 'A3': 1.3, 'Q3': 1.35,
    'XC40': 1.35, 'XC60': 1.45, 'XC90': 1.55, 'V60': 1.3, 'S60': 1.35,
    'NX': 1.45, 'RX': 1.55, 'ES': 1.5, 'IS': 1.4,
    'Stinger': 1.35, 'Mustang': 1.45, 'MX-5': 1.35,
    'Tiguan': 1.15, 'Passat': 1.12, 'Touareg': 1.4,
    # New brands
    'Corsa': 0.95, 'Astra': 1.0, 'Insignia': 1.05,
    '500': 1.1, 'Panda': 0.95, 'Tipo': 0.95,
    'Renegade': 1.15, 'Compass': 1.2, 'Cherokee': 1.35, 'Grand Cherokee': 1.45, 'Wrangler': 1.5,
    'Cooper': 1.25, 'Countryman': 1.35,
    '911': 2.0, 'Cayenne': 1.7, 'Macan': 1.6, 'Panamera': 1.8, 'Taycan': 1.9,
    'Model 3': 1.6, 'Model Y': 1.65, 'Model S': 1.8, 'Model X': 1.85,
    'Range Rover': 1.8, 'Range Rover Sport': 1.75, 'Discovery': 1.6, 'Defender': 1.7, 'Evoque': 1.5,
    'XE': 1.4, 'XF': 1.5, 'F-Pace': 1.55, 'E-Pace': 1.45, 'I-Pace': 1.65
}

cities = ['Tel Aviv', 'Jerusalem', 'Haifa', 'Rishon LeZion', 'Petah Tikva',
          'Ashdod', 'Netanya', 'Beer Sheva', 'Holon', 'Ramat Gan']

data = []

# Model-specific year ranges (for classic/discontinued models)
model_year_ranges = {
    '323F': (1994, 1998),  # Mazda 323F BA production years
}

# Generate 15000 sample records (LARGE dataset for better AI training)
for _ in range(15000):
    brand = np.random.choice(brands)
    model = np.random.choice(models_map[brand])

    # Use specific year range for certain models, otherwise default
    if model in model_year_ranges:
        year_min, year_max = model_year_ranges[model]
        year = np.random.randint(year_min, year_max + 1)
    else:
        year = np.random.randint(2010, 2025)

    # REALISTIC Base price based on Yad2 2024 prices
    # Formula: newer cars are significantly more expensive
    age = 2024 - year

    # Base price by age (more realistic depreciation)
    if age == 0:
        base_price = 120000  # New cars
    elif age == 1:
        base_price = 105000
    elif age == 2:
        base_price = 92000
    elif age == 3:
        base_price = 82000
    elif age == 4:
        base_price = 73000
    elif age <= 6:
        base_price = 60000
    elif age <= 8:
        base_price = 50000
    elif age <= 10:
        base_price = 42000
    elif age <= 15:
        base_price = 35000 - (age - 10) * 1000
    else:
        # Very old cars (20+ years) - minimum price floor
        # Classic/working cars still have value
        base_price = max(25000, 38000 - (age - 15) * 800)

    # Premium brands multiplier (based on real Yad2 prices)
    if brand in ['Porsche']:
        base_price *= 3.0  # Super premium
    elif brand in ['Tesla', 'Land Rover', 'Jaguar']:
        base_price *= 2.2  # Luxury/Electric
    elif brand in ['BMW', 'Mercedes', 'Audi', 'Lexus']:
        base_price *= 1.85  # Premium
    elif brand in ['Volvo']:
        base_price *= 1.5
    elif brand in ['Mini', 'Jeep']:
        base_price *= 1.3
    elif brand in ['Volkswagen', 'Subaru']:
        base_price *= 1.15
    elif brand in ['Mazda', 'Honda']:
        base_price *= 1.05
    elif brand in ['Opel', 'Fiat']:
        base_price *= 0.8  # Budget brands
    elif brand in ['Suzuki', 'Mitsubishi', 'Chevrolet']:
        base_price *= 0.85

    price_variation = np.random.randint(-8000, 8000)
    price = max(15000, base_price + price_variation)

    # Mileage depends on year (older = more km)
    age = 2024 - year
    if age == 0:
        mileage = np.random.randint(1000, 15000)
    elif age <= 2:
        mileage = np.random.randint(5000, 30000 * age)
    else:
        mileage = np.random.randint(10000 * age, 20000 * age + 50000)
    mileage = max(1000, min(400000, mileage))

    # Hand (owner number) - older cars tend to have more owners (up to 8)
    if age <= 2:
        hand = 1
    elif age <= 5:
        hand = np.random.choice([1, 2], p=[0.7, 0.3])
    elif age <= 10:
        hand = np.random.choice([1, 2, 3, 4], p=[0.2, 0.4, 0.25, 0.15])
    elif age <= 15:
        hand = np.random.choice([2, 3, 4, 5, 6], p=[0.1, 0.3, 0.3, 0.2, 0.1])
    else:
        # Very old cars (20+ years) can have many owners
        hand = np.random.choice([3, 4, 5, 6, 7, 8], p=[0.1, 0.2, 0.25, 0.25, 0.15, 0.05])

    engine = np.random.choice([1200, 1400, 1600, 1800, 2000, 2200, 2500, 3000, 3500])
    city = np.random.choice(cities)

    # Apply rarity multiplier
    rarity_multiplier = model_rarity.get(model, 1.0)
    price *= rarity_multiplier

    # REALISTIC mileage impact (based on Yad2 market data)
    # Average km per year: 15,000-20,000
    expected_mileage = age * 15000
    mileage_deviation = mileage - expected_mileage

    # Penalty for above-average mileage
    if mileage_deviation > 0:
        # Each extra 10,000 km reduces price by 3-5%
        mileage_penalty_percent = (mileage_deviation / 10000) * 0.035
        price *= (1 - mileage_penalty_percent)

    # Bonus for below-average mileage
    elif mileage_deviation < 0 and age > 0:
        mileage_bonus_percent = abs(mileage_deviation / 10000) * 0.02
        price *= (1 + min(mileage_bonus_percent, 0.15))  # Max 15% bonus

    # Additional strong penalty for very high mileage
    if mileage > 200000:
        price *= 0.75  # 25% penalty for 200k+ km
    elif mileage > 150000:
        price *= 0.85  # 15% penalty for 150k+ km

    # Hand penalty (realistic Yad2 impact - up to 8 hands)
    if hand == 1:
        pass  # No penalty
    elif hand == 2:
        price *= 0.93  # 7% reduction
    elif hand == 3:
        price *= 0.85  # 15% reduction
    elif hand == 4:
        price *= 0.75  # 25% reduction
    elif hand == 5:
        price *= 0.68  # 32% reduction
    elif hand == 6:
        price *= 0.62  # 38% reduction
    elif hand == 7:
        price *= 0.55  # 45% reduction
    else:  # hand >= 8
        price *= 0.50  # 50% reduction (very suspicious)

    # Minimum price floor - even old cars have value
    # Working cars can't be cheaper than scrap value
    if age > 20:
        price = max(18000, int(price))  # Old but working
    else:
        price = max(15000, int(price))

    data.append({
        'brand': brand,
        'model': model,
        'year': year,
        'price': price,
        'mileage_km': mileage,
        'hand': hand,
        'engine_capacity': engine,
        'city': city,
        'rarity_score': rarity_multiplier
    })

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('cars_data.csv', index=False, encoding='utf-8-sig')

print(f"Generated {len(df)} sample car records")
print("\nSample data:")
print(df.head(10))
print(f"\nData info:")
print(df.info())
print(f"\nStatistics:")
print(df.describe())
print(f"\nBrands distribution:")
print(df['brand'].value_counts())
