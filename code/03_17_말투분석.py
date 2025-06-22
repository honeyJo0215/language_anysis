import re
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans

# nltk의 VADER 사전 다운로드 (최초 1회 실행)
nltk.download('vader_lexicon')

# 한글 폰트 설정 (Windows의 경우 'Malgun Gothic' 권장)
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'Malgun Gothic'

# --- 1. 텍스트 전처리 및 특징 추출 함수 ---
def clean_text(text):
    """
    소문자화, URL 제거, 알파벳 및 공백만 남기도록 특수문자 제거
    """
    text = text.lower()
    text = re.sub(r'http\S+', '', text)  # URL 제거
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # 알파벳과 공백만 남김
    return text

def extract_features(message, sia):
    """
    메시지에서 VADER 감정 점수와 심리학적 특징(메시지 길이, 느낌표/물음표 개수 등)을 추출
    """
    sentiment_scores = sia.polarity_scores(message)
    features = {
        'sentiment_neg': sentiment_scores['neg'],
        'sentiment_neu': sentiment_scores['neu'],
        'sentiment_pos': sentiment_scores['pos'],
        'sentiment_compound': sentiment_scores['compound'],
        'message_length': len(message.split()),
        'exclamation_count': message.count('!'),
        'question_count': message.count('?')
    }
    return features

# --- 2. 사용자 입력: 원하는 만큼 메시지 입력 받기 ---
print("텍스트 메시지를 입력하세요. 입력을 마치려면 'quit'을 입력하세요.\n")
user_messages = []
while True:
    message = input("메시지를 입력하세요 (또는 'quit' 입력): ")
    if message.lower() == 'quit':
        break
    if message.strip() != "":
        user_messages.append(message)

if len(user_messages) == 0:
    print("입력된 메시지가 없습니다. 프로그램을 종료합니다.")
    exit()

# DataFrame 생성
df = pd.DataFrame({'message': user_messages})

# --- 3. 메시지별 특징 추출 ---
sia = SentimentIntensityAnalyzer()
features_list = df['message'].apply(lambda x: extract_features(x, sia))
features_df = pd.DataFrame(list(features_list))
df = pd.concat([df, features_df], axis=1)

# --- 4. (선택사항) 클러스터링을 통한 메시지 패턴 그룹화 ---
numeric_features = df[['sentiment_compound', 'message_length', 'exclamation_count', 'question_count']].values
kmeans = KMeans(n_clusters=2, random_state=42)
df['cluster'] = kmeans.fit_predict(numeric_features)

# --- 5. 전체 스타일 요약 통계 계산 ---
summary_stats = df[['sentiment_compound', 'message_length', 'exclamation_count', 'question_count']].mean()

print("\n전체 메시지 스타일 요약:")
print(summary_stats)

# --- 6. 단어 사용 빈도 분석 (불용어 제거) ---
vectorizer = CountVectorizer(stop_words='english')
word_matrix = vectorizer.fit_transform(df['message'])
word_counts = np.array(word_matrix.sum(axis=0)).flatten()
word_freq = dict(zip(vectorizer.get_feature_names_out(), word_counts))

# 상위 10개 단어 추출
top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
print("\n상위 10개 단어 빈도:")
for word, count in top_words:
    print(f"{word}: {count}")

# --- 7. 분석 결과에 따른 텍스트 스타일 가이드 제공 ---
print("\n분석 결과에 따른 텍스트 스타일 가이드:")
print("1. 긍정적 정서 표현이 두드러지며, 평균 compound 점수는 {:.2f}입니다.".format(summary_stats['sentiment_compound']))
print("2. 평균 메시지 길이는 {:.1f} 단어로, 비교적 간결한 문장을 사용하는 경향이 있습니다.".format(summary_stats['message_length']))
print("3. 느낌표와 물음표 사용 빈도가 감정을 강조하는 스타일을 보여줍니다.")
print("4. 자주 사용되는 단어: " + ", ".join([word for word, count in top_words]) + " 등이 관찰됩니다.")
print("\n이 가이드를 참고하여 상대의 말투와 텍스트 습관을 모방해 보세요.")

# --- 8. 시각화 ---
# (1) 클러스터링 결과 시각화: 메시지 길이 vs. compound 정서 점수
plt.figure(figsize=(8,6))
plt.scatter(df['message_length'], df['sentiment_compound'], c=df['cluster'], cmap='viridis', s=100, alpha=0.7)
plt.xlabel("메시지 길이 (단어 수)")
plt.ylabel("Compound 정서 점수")
plt.title("메시지 패턴 클러스터링 결과")
plt.grid(True)
plt.show()

# (2) 메시지 길이 분포 히스토그램
plt.figure(figsize=(8,6))
plt.hist(df['message_length'], bins=10, alpha=0.7)
plt.xlabel("메시지 길이 (단어 수)")
plt.ylabel("빈도")
plt.title("메시지 길이 분포")
plt.show()

# (3) 정서 점수 분포 히스토그램
plt.figure(figsize=(8,6))
plt.hist(df['sentiment_compound'], bins=10, alpha=0.7)
plt.xlabel("Compound 정서 점수")
plt.ylabel("빈도")
plt.title("정서 점수 분포")
plt.show()
