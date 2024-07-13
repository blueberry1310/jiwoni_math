import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

# 데이터 로드
file_path = 'survey_data.csv'  # 원본 설문 데이터 파일 경로
data = pd.read_csv(file_path)

# 결측치 처리 (예: 결측치는 빈 문자열로 대체)
data.fillna('', inplace=True)

# 만족도 데이터 인코딩
satisfaction_mapping = {'매우 만족': 5, '만족': 4, '보통': 3, '불만족': 2, '매우 불만족': 1}
data.replace(satisfaction_mapping, inplace=True)

# '건의 사항' 관련 열들을 하나로 합침
data['통합 건의사항'] = data[['체육관 관련 건의 사항(필수X)', '화장실 관련 건의 사항(필수X)', '매점 건의 사항(필수X)', '자습실 건의 사항(필수X)']].apply(lambda x: ' '.join(x), axis=1)

# 텍스트 데이터 벡터화
vectorizer = TfidfVectorizer(max_features=100)  # 필요한 경우 max_features 값을 조정하세요
text_features = vectorizer.fit_transform(data['통합 건의사항']).toarray()

# 텍스트 피처 데이터프레임으로 변환
text_features_df = pd.DataFrame(text_features, columns=vectorizer.get_feature_names_out())

# 피처 데이터프레임에서 만족도 열과 통합 건의사항 열을 제외한 나머지 피처 선택
features = data[['체육관 장비(농구공, 배드민턴 라켓 등) 상태는 어떤가요?', '체육관의 청결은 어떤가요?', '체육관의 에어컨 온도는 어떤가요?', '화장실 관리 비품(휴지, 비누 등) 리필 상태는 어떤가요?', '화장실 청결 상태는 어떤가요?', '매점의 상품 다양성은 어떤가요?', '매점의 상품 가격은 어떤가요?', '자습실의 소음관리는\n어떤가요?', '자습실의 좌석 수는\n어떤가요?', '자습실의 청결 상태는\n어떤가요?']]

# 문자열 응답을 숫자로 변환 (필요한 경우, 특정 문자열 응답을 숫자로 매핑)
replacement_dict = {'사용해본 적이 없어 모르겠다': 0, '': 0}
features.replace(replacement_dict, inplace=True)

# 텍스트 피처와 나머지 피처를 병합
features = pd.concat([features, text_features_df], axis=1)

# 레이블 열 선택 (여기서는 예시로 '체육관 장비(농구공, 배드민턴 라켓 등) 상태는 어떤가요?'를 사용)
labels = data['체육관 장비(농구공, 배드민턴 라켓 등) 상태는 어떤가요?']

# 데이터 정규화
scaler = StandardScaler()
features = scaler.fit_transform(features)

# 피처와 레이블을 데이터프레임으로 병합
processed_data = pd.DataFrame(features)
processed_data['체육관 장비 상태 만족도'] = labels.values

# 전처리된 데이터를 새로운 CSV 파일로 저장
processed_data.to_csv('processed_survey_data.csv', index=False)

print('데이터 전처리 완료 및 저장 완료')

