from .db import Base, engine, SessionLocal
from .models import Character
from .vectorstore import add_or_update

# minimal demo properties
PROPS = [
    {
        "id": "D7-Apt-2BR-900",
        "description": "Cozy 2BR apartment in District 7, 65sqm, near Crescent Mall.",
        "design_and_layout": {"location": "District 7", "type": "apartment", "bedrooms": 2, "price": 900},
        "physical_features": {},
        "property_groups": [],
        "_meta": {"title":"D7 Apartment", "location":"District 7", "property_type":"apartment", "bedrooms":2, "price":900, "description":"Cozy 2BR"}
    },
    {
        "id": "ThuDuc-Condo-1BR-600",
        "description": "Modern 1BR condo in Thu Duc City, 48sqm, high floor.",
        "design_and_layout": {"location": "Thu Duc", "type": "condo", "bedrooms": 1, "price": 600},
        "physical_features": {},
        "property_groups": [],
        "_meta": {"title":"Thu Duc Condo", "location":"Thu Duc", "property_type":"condo", "bedrooms":1, "price":600, "description":"Modern 1BR"}
    },
]

Base.metadata.create_all(bind=engine)

def main():
    with SessionLocal() as db:
        if not db.query(Character).first():
            db.add(Character(name="Ava", style="Warm, concise, proactive", objective="Help users find best-fit properties and schedule viewings"))
            db.commit()

    for d in PROPS:
        add_or_update(d)
    print("Seeded characters + embedded demo properties.")

if __name__ == "__main__":
    main()