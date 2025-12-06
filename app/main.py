from fastapi import FastAPI
from app.loaders import load_multipleye_data

app = FastAPI()


@app.get("/load-data")
def load_data():
    df_ro, df_en = load_multipleye_data()

    return {
        "romanian": {
            "count": len(df_ro),
            "sample": df_ro.head(1).to_dict(orient="records")
        },
        "english": {
            "count": len(df_en),
            "sample": df_en.head(1).to_dict(orient="records")
        }
    }
