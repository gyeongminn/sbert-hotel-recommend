from sentence_transformers import SentenceTransformer, util

# 문장 임베딩 모델 https://github.com/snunlp/KR-SBERT
model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

# 예시 입력 문장
input_sentence = "깔끔하고 저렴한 호텔"

# 예시 데이터
hotel_reviews = {
    "더 스테이트 선유": ["1박 금액도 저렴한데 굉장히 깔끔하고 1층에 카페도 있어 좋았습니다.", "선유도역에서도 가깝고 출장으로 자주 이용하고 있습니다~깨끗하고 난방도 잘되고 좋아요", "지하철 역 근처여서 소음이 있을 줄 알았는데 방음이 잘되서 조용하게 잘 쉴 수 있었어요 !!"],
    "롯데시티호텔 명동": ["좋았어요!!너무 좋은 추억 남기고 갑니당!", "직원분들친절하고 내부 깔끔하고 좋았습니다.", "출장으로 편하게 쉬고 왔어요", "명둉성당 가깝고 청계천도 가까워요","청결하고 조용하고 좋았음"],
    "글래드 마포": ["직원분들이 모두 친절하셨고 시설도 깔끔했어요! 편안히 묵고 갑니다 감사합니다:)", "직원분들도 친절하시고 룸 컨디션도 좋습니다.", "모두 좋은데 대실시간이 너무 타잇트했어요 4시 무조건 딱 나가야하는 ㅠ"]
}

# 모든 리뷰 벡터화 및 유사도 계산
all_scores = []
for hotel_name, reviews in hotel_reviews.items():
    review_embeddings = model.encode(reviews, convert_to_tensor=True)
    sentence_embedding = model.encode(input_sentence, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(sentence_embedding, review_embeddings)[0]

    for score in cosine_scores:
        all_scores.append((hotel_name, score.item()))

# 상위 3개 결과 추출
all_scores.sort(key=lambda x: x[1], reverse=True)
top_3 = all_scores[:3]

# 결과 출력
print("Input:", input_sentence)
print("\nTop 3 Similar Hotels:")
for hotel_name, score in top_3:
    print(f"호텔: {hotel_name}, 유사도: {score:.4f}")
