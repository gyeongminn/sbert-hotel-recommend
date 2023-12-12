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

# 호텔별 리뷰 통합 및 벡터화
hotel_embeddings = {}
for hotel_name, reviews in hotel_reviews.items():
    combined_reviews = " ".join(reviews)  # 모든 리뷰를 하나의 텍스트로 통합
    hotel_embeddings[hotel_name] = model.encode(combined_reviews, convert_to_tensor=True)

# 입력 문장의 벡터화
sentence_embedding = model.encode(input_sentence, convert_to_tensor=True)

# 호텔별 유사도 계산
similarity_scores = []
for hotel_name, embedding in hotel_embeddings.items():
    cosine_score = util.pytorch_cos_sim(sentence_embedding, embedding)[0].item()
    similarity_scores.append((hotel_name, cosine_score))

# 상위 3개 결과 추출 및 정렬
top_3_hotels = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[:3]

# 결과 출력
print("Input:", input_sentence)
print("\nTop 3 Similar Hotels:")
for hotel_name, score in top_3_hotels:
    print(f"호텔: {hotel_name}, 유사도: {score:.4f}")