from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

# Inisialisasi Flask
app = Flask(__name__)

# 1. Memuat data
cellphones = pd.read_csv('cellphones data.csv')
ratings = pd.read_csv('cellphones ratings.csv')

# 2. Membaca data ke dalam format Surprise
reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(ratings[['user_id', 'cellphone_id', 'rating']], reader)

# 3. Membagi data menjadi train dan test
trainset, testset = train_test_split(data, test_size=0.2)

# 4. Menggunakan SVD untuk collaborative filtering
model = SVD()
model.fit(trainset)

# 5. Fungsi untuk mendapatkan rekomendasi
def get_recommendations(user_rating, num_recommendations=5):
    # Memprediksi skor untuk semua ponsel
    all_cellphones = cellphones['cellphone_id'].unique()
    predictions = [(cellphone_id, user_rating) for cellphone_id in all_cellphones]

    # Mengurutkan berdasarkan skor
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_recommendations = predictions[:num_recommendations]

    # Mengambil detail ponsel
    recommended_cellphones = cellphones[cellphones['cellphone_id'].isin([x[0] for x in top_recommendations])]
    return recommended_cellphones

# 6. Rute Flask
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Ambil rating dari pengguna
        user_rating = int(request.form["rating"])
        
        # Redirect ke halaman rekomendasi
        return redirect(url_for("recommendations", rating=user_rating))
    
    return render_template("index.html")

@app.route("/recommendations/<int:rating>")
def recommendations(rating):
    # Dapatkan rekomendasi berdasarkan rating pengguna
    recommended_cellphones = get_recommendations(rating)
    return render_template("recommendations.html", cellphones=recommended_cellphones)

# Menjalankan aplikasi
if __name__ == "__main__":
    app.run(debug=True)
