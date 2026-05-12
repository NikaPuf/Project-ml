import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import re

def get_image_path(base_path):
    for ext in ['.png', '.jpg', '.jpeg']:
        if os.path.exists(base_path + ext):
            return base_path + ext
    return None

@st.cache_resource
def load_ml_components():
    model_path = 'models/best_model.pkl'
    scaler_path = 'models/scaler.pkl'
    le_country_path = 'models/le_country.pkl'
    le_major_path = 'models/le_major.pkl'
    le_target_uni_path = 'models/le_target_uni.pkl'
    uni_data_path = 'universities_data.csv'

    for path in [model_path, scaler_path, le_country_path, le_major_path, le_target_uni_path, uni_data_path]:
        if not os.path.exists(path):
            st.error(f"Файл {path} не найден!")
            st.stop()

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    le_country = joblib.load(le_country_path)
    le_major = joblib.load(le_major_path)
    le_target_uni = joblib.load(le_target_uni_path)
    uni_df = pd.read_csv(uni_data_path)

    uni_df.columns = uni_df.columns.str.lower()

    if 'name' in uni_df.columns:
        uni_df['name'] = uni_df['name'].str.replace('New York Universuty (NYU)', 'New York University (NYU)')
        uni_df['name'] = uni_df['name'].str.replace('University Colledge London', 'University College London')

    if 'country' not in uni_df.columns:
        for col in uni_df.columns:
            if 'country' in col.lower():
                uni_df.rename(columns={col: 'country'}, inplace=True)
                break
        else:
            st.error("Колонка 'country' не найдена!")
            st.stop()

    for col in ['other_exam_type', 'other_exam_score']:
        if col in uni_df.columns:
            uni_df[col] = uni_df[col].apply(lambda x: None if pd.isna(x) or x == 'nan' or str(x).strip() == '' else x)

    region_map = {
        'usa': 'USA',
        'united kingdom': 'UK',
        'china': 'China',
        'switzerland': 'Europe',
        'singapore': 'Asia',
        'canada': 'USA'
    }

    uni_df['country_lower'] = uni_df['country'].str.lower()
    uni_df['region'] = uni_df['country_lower'].map(region_map)

    def format_percent(value):
        if pd.isna(value):
            return 'данные отсутствуют'
        if isinstance(value, str) and '%' in value:
            return value
        if isinstance(value, (int, float)):
            if value < 1:
                return f"{int(round(value * 100))}%"
            else:
                return f"{int(value)}%"
        if isinstance(value, str):
            numbers = re.findall(r'(\d+\.?\d*)', value)
            if numbers:
                num = float(numbers[0])
                if num < 1:
                    return f"{int(round(num * 100))}%"
                else:
                    return f"{int(num)}%"
        return 'данные отсутствуют'

    percent_col = None
    for col in uni_df.columns:
        if 'percentage' in col.lower() or 'foreign' in col.lower():
            percent_col = col
            break

    if percent_col:
        uni_df['foreign_percent_display'] = uni_df[percent_col].apply(format_percent)
    else:
        uni_df['foreign_percent_display'] = 'данные отсутствуют'

    return model, scaler, le_country, le_major, le_target_uni, uni_df

def prepare_features(student, uni_row, le_country, le_major, le_target_uni):
    features = {}

    features['gpa'] = student['gpa']
    features['target_gpa'] = uni_row['gpa_requirement']
    features['gpa_diff'] = features['gpa'] - features['target_gpa']
    features['gpa_meets_requirement'] = 1 if features['gpa'] >= features['target_gpa'] else 0

    features['exam_type'] = student['exam_type']
    features['target_exam_type'] = uni_row['exam_type']
    features['target_exam_score'] = uni_row['exam_score']
    features['exam_score'] = student['exam_score']

    if features['exam_type'] == features['target_exam_type'] and features['exam_score'] is not None:
        features['exam_meets_requirement'] = 1 if features['exam_score'] >= features['target_exam_score'] else 0
        features['exam_above_requirement'] = features['exam_score'] - features['target_exam_score']
    else:
        features['exam_meets_requirement'] = 0
        features['exam_above_requirement'] = 0

    features['other_exam_type'] = student.get('other_exam_type')
    features['target_other_exam_type'] = uni_row.get('other_exam_type')
    features['target_other_exam_score'] = uni_row.get('other_exam_score')
    features['other_exam_score'] = student.get('other_exam_score')

    if (features['other_exam_type'] == features['target_other_exam_type'] and
            features['other_exam_score'] is not None and
            features['target_other_exam_type'] is not None):
        features['other_exam_meets_requirement'] = 1 if features['other_exam_score'] >= features['target_other_exam_score'] else 0
        features['other_exam_above_requirement'] = features['other_exam_score'] - features['target_other_exam_score']
    else:
        features['other_exam_meets_requirement'] = 0
        features['other_exam_above_requirement'] = 0

    volunteer_hours = student.get('volunteer_score', 0)
    features['experience_score'] = volunteer_hours / 10

    try:
        features['country_encoded'] = le_country.transform([student['country']])[0]
    except ValueError:
        features['country_encoded'] = 0

    major = student.get('major', 'Computer Science')
    try:
        features['major_encoded'] = le_major.transform([major])[0]
    except ValueError:
        features['major_encoded'] = 0

    target_uni = student['target_university']
    try:
        features['target_uni_encoded'] = le_target_uni.transform([target_uni])[0]
    except ValueError:
        features['target_uni_encoded'] = 0

    feature_order = [
        'gpa', 'gpa_diff', 'gpa_meets_requirement',
        'exam_meets_requirement', 'exam_above_requirement',
        'other_exam_meets_requirement', 'other_exam_above_requirement',
        'experience_score',
        'country_encoded', 'major_encoded', 'target_uni_encoded'
    ]

    X = np.array([[features.get(f, 0) for f in feature_order]], dtype=float)
    return X

def predict_single(student, model, scaler, le_country, le_major, le_target_uni, uni_df):
    uni_row = uni_df[uni_df['name'] == student['target_university']].iloc[0]
    X = prepare_features(student, uni_row, le_country, le_major, le_target_uni)
    X_scaled = scaler.transform(X)
    proba = model.predict_proba(X_scaled)[0, 1]
    return proba

def create_checklist(student, uni_row, prob_percent, is_china):
    checklist = []

    if student['gpa'] >= uni_row['gpa_requirement']:
        checklist.append({"task": f"GPA {student['gpa']} (требование {uni_row['gpa_requirement']})", "status": "Выполнено"})
    else:
        checklist.append({"task": f"Поднять GPA с {student['gpa']} до {uni_row['gpa_requirement']}", "status": "Требуется работа"})

    if student['exam_score'] is not None:
        if student['exam_type'] == uni_row['exam_type']:
            if student['exam_score'] >= uni_row['exam_score']:
                checklist.append({"task": f"{uni_row['exam_type']} {student['exam_score']} (требование {uni_row['exam_score']})", "status": "Выполнено"})
            else:
                checklist.append({"task": f"Поднять {uni_row['exam_type']} с {student['exam_score']} до {uni_row['exam_score']}", "status": "Требуется работа"})
        else:
            checklist.append({"task": f"Сдать {uni_row['exam_type']} (нужен балл {uni_row['exam_score']})", "status": "Не тот экзамен"})
    else:
        checklist.append({"task": f"Сдать {uni_row['exam_type']} (нужен балл {uni_row['exam_score']})", "status": "Требуется сдача"})

    if not is_china and uni_row.get('other_exam_type'):
        other_exam_score = student.get('other_exam_score')
        if other_exam_score and other_exam_score >= uni_row['other_exam_score']:
            checklist.append({"task": f"{uni_row['other_exam_type']} {other_exam_score} (требование {uni_row['other_exam_score']})", "status": "Выполнено"})
        elif other_exam_score:
            checklist.append({"task": f"Поднять {uni_row['other_exam_type']} с {other_exam_score} до {uni_row['other_exam_score']}", "status": "Требуется работа"})
        else:
            checklist.append({"task": f"Сдать {uni_row['other_exam_type']} (нужен балл {uni_row['other_exam_score']})", "status": "Требуется сдача"})

    volunteer_hours = student.get('volunteer_score', 0)
    if volunteer_hours >= 100:
        checklist.append({"task": f"Волонтерство: {volunteer_hours} часов", "status": "Выполнено"})
    elif volunteer_hours > 0:
        checklist.append({"task": f"Волонтерство: {volunteer_hours} часов (рекомендуется 100+)", "status": "Можно улучшить"})
    else:
        checklist.append({"task": "Волонтерство (рекомендуется 100+ часов)", "status": "Требуется добавить"})

    if prob_percent >= 80:
        checklist.append({"task": "Подача документов", "status": "Рекомендуем подавать"})
        checklist.append({"task": "Мотивационное письмо", "status": "Напишите сильное письмо"})
        checklist.append({"task": "Рекомендательные письма", "status": "Запросите у преподавателей"})
    elif prob_percent >= 50:
        checklist.append({"task": "Подача документов", "status": "Стоит попробовать"})
        checklist.append({"task": "Запасной вариант", "status": "Выберите еще 2-3 вуза"})
        checklist.append({"task": "Усиление портфолио", "status": "Добавьте проекты"})
    else:
        checklist.append({"task": "Подача документов", "status": "Шансы низкие"})
        checklist.append({"task": "Другие вузы", "status": "Рассмотрите альтернативы"})
        checklist.append({"task": "Улучшение профиля", "status": "Повысьте GPA или пересдайте экзамен"})

    return checklist

st.set_page_config(page_title="Global Admission Assistant", layout="wide")

st.markdown("""
<style>
    .stApp { background: #f5f5f5; }
    .main-title {
        font-family: 'Georgia', serif;
        font-size: 32px;
        font-weight: 600;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 5px;
    }
    .sub-title {
        font-family: 'Georgia', serif;
        font-size: 14px;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 30px;
        padding-bottom: 15px;
    }
    .form-card {
        background: white;
        padding: 25px;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
    }
    .form-group { margin-bottom: 20px; }
    .form-group label {
        display: block;
        margin-bottom: 8px;
        font-weight: 600;
        color: #2c3e50;
        font-family: 'Segoe UI', sans-serif;
        font-size: 14px;
    }
    .form-hint {
        font-size: 12px;
        color: #7f8c8d;
        margin-top: 4px;
        font-family: 'Segoe UI', sans-serif;
    }
    .uni-card-usa, .uni-card-china, .uni-card-uk, .uni-card-europe, .uni-card-asia {
        background: white;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
        font-family: 'Segoe UI', sans-serif;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border-top: 1px solid;
        border-right: 1px solid;
        border-bottom: 1px solid #e0e0e0;
        border-left: 4px solid;
    }
    .uni-card-china {
        border-left-color: #de2910;
        border-top-color: #ffcc00;
        border-right-color: #ffcc00;
    }
    .uni-card-usa, .uni-card-uk, .uni-card-europe, .uni-card-asia {
        border-left-color: #b22234;
        border-top-color: #3c3b6e;
        border-right-color: #3c3b6e;
    }
    .uni-card h3 {
        margin: 0 0 15px 0;
        color: #2c3e50;
        font-size: 20px;
        font-weight: 600;
        font-family: 'Georgia', serif;
    }
    .uni-card p { margin: 8px 0; color: #34495e; font-size: 14px; }
    .uni-card .requirement { font-weight: 600; }
    .result-card {
        background: white;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin-top: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .result-card h4 {
        margin: 0 0 15px 0;
        color: #2c3e50;
        font-size: 18px;
        font-weight: 600;
        font-family: 'Georgia', serif;
        border-bottom: 2px solid #3498db;
        padding-bottom: 8px;
    }
    .probability-text { font-size: 28px; font-weight: 700; margin: 15px 0; }
    .checklist-item {
        padding: 10px 12px;
        margin: 8px 0;
        border-radius: 6px;
        background: #f9f9f9;
        border-left: 3px solid #3498db;
        font-family: 'Segoe UI', sans-serif;
        font-size: 14px;
    }
    .stButton button {
        background-color: #3498db;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 6px;
        cursor: pointer;
        font-family: 'Segoe UI', sans-serif;
        font-weight: 600;
        font-size: 14px;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton button:hover { background-color: #2980b9; color: white; }
    .stRadio > div { flex-direction: row; gap: 20px; font-family: 'Segoe UI', sans-serif; }
    .stSelectbox label, .stSlider label, .stTextInput label, .stRadio label, .stCheckbox label {
        font-family: 'Segoe UI', sans-serif;
        font-weight: 600;
        color: #2c3e50;
    }
    .stSlider {
        background: transparent !important;
        box-shadow: none !important;
    }
    div[data-baseweb="slider"] {
        background: transparent !important;
        height: 6px;
    }
    div[data-baseweb="slider"] > div {
        background: #e0e0e0 !important;
    }
    div[data-testid="stSlider"] div[data-baseweb="slider"] div[role="slider"] {
        background-color: #3498db;
        border: none;
        width: 16px;
        height: 16px;
    }
    .uni-image { margin-top: 15px; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
    @media (max-width: 768px) {
        .main-title { font-size: 24px; }
        .form-card { padding: 15px; }
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Global Admission Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Оценка вероятности поступления в зарубежные университеты</div>', unsafe_allow_html=True)

with st.spinner("Загрузка..."):
    model, scaler, le_country, le_major, le_target_uni, uni_df = load_ml_components()

left_col, right_col = st.columns([2, 1])

with left_col:
    st.markdown('<div class="form-card">', unsafe_allow_html=True)

    uni_names = sorted(uni_df['name'].tolist())
    university = st.selectbox("Выберите университет", ["— Выберите вуз —"] + uni_names)

    is_china = False
    if university != "— Выберите вуз —":
        uni_info = uni_df[uni_df['name'] == university].iloc[0]
        is_china = (uni_info['country'] == 'China')

    st.markdown('<label>Средний балл (GPA)</label>', unsafe_allow_html=True)
    gpa = st.slider("gpa_slider", min_value=2.0, max_value=4.0, value=3.5, step=0.1, label_visibility="collapsed")
    st.markdown(f'<div style="text-align: center; font-size: 18px; color: #3498db; font-weight: bold; margin-top: 8px;">{gpa}</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size: 12px; color: #7f8c8d; margin-top: 4px;">Укажите ваш средний балл по 4-балльной шкале</div>', unsafe_allow_html=True)

    st.markdown('<div class="form-group">', unsafe_allow_html=True)
    st.markdown('<label>Тип языкового экзамена</label>', unsafe_allow_html=True)
    if is_china:
        exam_options = ["HSK (Китайский)"]
    else:
        exam_options = ["IELTS (Международный)"]
    exam = st.radio("exam_radio", exam_options, horizontal=True, label_visibility="collapsed")
    exam_type = "HSK" if "HSK" in exam else "IELTS"
    st.markdown('<div class="form-hint">Выберите экзамен, который вы сдаёте или планируете сдавать</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if exam_type == "IELTS":
        exam_score_str = st.text_input("Балл IELTS", placeholder="IELTS: 6.5, 7.0, 7.5...")
    else:
        exam_score_str = st.text_input("Уровень HSK", placeholder="HSK: 4, 5, 6")
    exam_score = None
    if exam_score_str:
        try:
            exam_score = float(exam_score_str) if exam_type == "IELTS" else int(exam_score_str)
        except:
            st.warning("Введите корректное число")

    other_exam_type = None
    other_exam_score = None

    if not is_china and university != "— Выберите вуз —":
        st.markdown('<div class="form-group">', unsafe_allow_html=True)
        st.markdown('<label>Дополнительный экзамен</label>', unsafe_allow_html=True)

        target_other_exam = uni_info.get('other_exam_type')
        if target_other_exam == 'SAT':
            other_exam_options = ["SAT", "Не сдаю"]
            other_exam = st.radio("other_exam_radio", other_exam_options, horizontal=True, label_visibility="collapsed")
            if other_exam == "SAT":
                other_exam_type = "SAT"
                other_exam_score_str = st.text_input("Балл SAT", placeholder="SAT: 1200-1600")
                if other_exam_score_str:
                    try:
                        other_exam_score = int(other_exam_score_str)
                    except:
                        st.warning("Введите корректное число")
        elif target_other_exam == 'IB':
            other_exam_options = ["IB", "Не сдаю"]
            other_exam = st.radio("other_exam_radio", other_exam_options, horizontal=True, label_visibility="collapsed")
            if other_exam == "IB":
                other_exam_type = "IB"
                other_exam_score_str = st.text_input("Балл IB", placeholder="IB: 30-45")
                if other_exam_score_str:
                    try:
                        other_exam_score = int(other_exam_score_str)
                    except:
                        st.warning("Введите корректное число")

        st.markdown('<div class="form-hint">SAT/IB требуются для поступления в западные университеты</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="form-group">', unsafe_allow_html=True)
    st.markdown('<label>Волонтерство</label>', unsafe_allow_html=True)
    volunteer_options = ["Нет", "Менее 50 часов", "50-100 часов", "100+ часов"]
    volunteer_map = {"Нет": 0, "Менее 50 часов": 25, "50-100 часов": 75, "100+ часов": 120}
    volunteer_choice = st.radio("volunteer_radio", volunteer_options, horizontal=True, label_visibility="collapsed")
    volunteer_hours = volunteer_map[volunteer_choice]
    st.markdown('<div class="form-hint">Волонтерский опыт повышает шансы на поступление</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    if university != "— Выберите вуз —":
        student = {
            'country': uni_info['country'],
            'major': 'Computer Science',
            'gpa': gpa,
            'exam_type': exam_type,
            'exam_score': exam_score,
            'other_exam_type': other_exam_type,
            'other_exam_score': other_exam_score,
            'volunteer_score': volunteer_hours,
            'target_university': university
        }

        if st.button("Рассчитать вероятность поступления"):
            if exam_score is None:
                st.warning(f"Укажите балл {exam_type}")
            else:
                prob = predict_single(student, model, scaler, le_country, le_major, le_target_uni, uni_df)
                prob_percent = int(round(prob * 100))
                checklist = create_checklist(student, uni_info, prob_percent, is_china)

                if prob_percent >= 80:
                    verdict = "Высокая вероятность поступления"
                    verdict_color = "#27ae60"
                elif prob_percent >= 50:
                    verdict = "Средняя вероятность поступления"
                    verdict_color = "#f39c12"
                else:
                    verdict = "Низкая вероятность поступления"
                    verdict_color = "#e74c3c"

                st.markdown(f"""
                <div class="result-card">
                    <h4>Результат оценки</h4>
                    <div class="probability-text" style="color: {verdict_color};">Вероятность поступления: {prob_percent}%</div>
                    <p style="color: {verdict_color}; font-weight: 600;">{verdict}</p>
                </div>
                <div class="result-card">
                    <h4>Выполнение требований</h4>
                """, unsafe_allow_html=True)

                for item in checklist:
                    if "Выполнено" in item["status"]:
                        status_icon = "✓"
                    elif "Требуется" in item["status"]:
                        status_icon = "○"
                    else:
                        status_icon = "⚠"
                    st.markdown(f'<div class="checklist-item">{status_icon} {item["task"]} — <strong>{item["status"]}</strong></div>', unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    if university != "— Выберите вуз —":
        country = uni_info['country']
        if country == 'China' or university == "Nanyang Technological University":
            card_class = "uni-card-china"
            main_color = "#de2910"
            accent_color = "#ffcc00"
        else:
            card_class = "uni-card-usa"
            main_color = "#b22234"
            accent_color = "#3c3b6e"

        foreign_display = uni_info.get('foreign_percent_display', 'данные отсутствуют')

        other_exam_text = ""
        other_exam_type_val = uni_info.get('other_exam_type')
        other_exam_score_val = uni_info.get('other_exam_score')
        if other_exam_type_val and other_exam_score_val and str(other_exam_score_val) != 'nan':
            other_exam_text = f"<p>• Дополнительный экзамен: <strong>{other_exam_type_val} {int(other_exam_score_val) if isinstance(other_exam_score_val, float) else other_exam_score_val}</strong></p>"
        else:
            other_exam_text = "<p>• Дополнительный экзамен: <strong>не требуется</strong></p>"

        st.markdown(f"""
        <div class="{card_class}">
            <h3>{university}</h3>
            <p><span class="requirement" style="color: {main_color};">Требования к поступающим:</span></p>
            <p>• Средний балл (GPA): <strong>{uni_info['gpa_requirement']}</strong> из 4.0</p>
            <p>• Языковой экзамен: <strong style="color: {accent_color};">{uni_info['exam_type']} {uni_info['exam_score']}</strong></p>
            {other_exam_text}
            <p>• Стоимость обучения: <strong>${uni_info.get('cost_value', 'по запросу')}</strong> в год</p>
            <p>• Доля иностранных студентов: <strong>{foreign_display}</strong></p>
        </div>
        """, unsafe_allow_html=True)

        found_key = None

        if "ETH" in university or "Zurich" in university:
            found_key = "eth_zurich"

        if found_key is None:
            if "MIT" in university:
                found_key = "mit"
            elif "Stanford" in university:
                found_key = "stanford"
            elif "Harvard" in university:
                found_key = "harvard"
            elif "NYU" in university or "New York" in university:
                found_key = "nyu"
            elif "Princeton" in university:
                found_key = "princeton"
            elif "Duke" in university:
                found_key = "duke"
            elif "Yale" in university:
                found_key = "yale"
            elif "Oxford" in university:
                found_key = "oxford"
            elif "Cambridge" in university:
                found_key = "cambridge"
            elif "Edinburgh" in university:
                found_key = "edinburgh"
            elif "Imperial" in university:
                found_key = "imperial"
            elif "College London" in university or "UCL" in university:
                found_key = "ucl"
            elif "Tsinghua" in university:
                found_key = "tsinghua"
            elif "Fudan" in university:
                found_key = "fudan"
            elif "Peking" in university:
                found_key = "peking"
            elif "Shanghai" in university:
                found_key = "shanghai_jiaotong"
            elif "Nanyang" in university or "NTU" in university:
                found_key = "ntu"
            elif "Singapore" in university or "NUS" in university:
                found_key = "nus"
            elif "Toronto" in university:
                found_key = "toronto"

        image_path = None
        if found_key:
            for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG']:
                test_path = f"images/{found_key}{ext}"
                if os.path.exists(test_path):
                    image_path = test_path
                    break

        if image_path:
            st.image(image_path, use_container_width=True)

    else:
        st.markdown(f"""
        <div class="uni-card-usa" style="border-left-color: #3498db;">
            <h3>Информация об университете</h3>
            <p>Выберите университет из списка слева, чтобы увидеть его требования и рекомендации.</p>
            <hr>
            <p style="font-size: 13px; color: #7f8c8d;">Доступно {len(uni_names)} университетов</p>
        </div>
        """, unsafe_allow_html=True)