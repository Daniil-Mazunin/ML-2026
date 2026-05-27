import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tensorflow.keras.models import load_model
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Inference Dashboard", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv('data/house_regression.csv')

df = load_data()

st.sidebar.title("Навигация")
page = st.sidebar.radio("Выберите страницу:", 
                        ["Информация о разработчике", 
                         "Информация о наборе данных", 
                         "Визуализация данных", 
                         "Предсказание моделей ML"])

if page == "Информация о разработчике":
    st.title("Расчетно-графическая работа по дисциплине «Машинное обучение и большие данные»")
    st.subheader("Информация о разработчике моделей ML")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("photo.jpg", width=260)
    
    with col2:
        st.write("**ФИО:** Мазунин Даниил Станиславович")
        st.write("**Учебная группа:** ФИТ-242")
        st.write("**Тема РГР:** Разработка Web-приложения (дашборда) для инференса (вывода) моделей ML и анализа данных")

elif page == "Информация о наборе данных":
    st.title("Описание набора данных и предобработка")
    st.write('Этот набор данных содержит информацию о ценах на продажу домов в округе Кинг, в который входит Сиэтл. В него входят дома, проданные в период с мая 2014 по май 2015 года.')

    data = pd.read_csv('data/kc_house_data.csv')

    st.write("### Первые строки датасета")
    st.dataframe(data.head()) 
        
    st.write("### Предобработка данных")
    st.write("Было произведено извлечение года и месяца из даты, расчет возраста дома, логарифмирование признаков с площадью и удаление лишних признаков.")
        
    data['date'] = pd.to_datetime(data['date'], format='%Y%m%dT%H%M%S')
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['house_age'] = data['year'] - data['yr_built']
    data = data.drop(columns=['id', 'date', 'yr_built', 'zipcode']) 
        
    columns = ['bedrooms', 'bathrooms', 'waterfront', 'view', 'condition', 'grade', 'floors', 'month', 'house_age']
    data[columns] = data[columns].astype('int8')
    data['year'] = data['year'].astype('int32') 
        
    st.write("### Описательная статистика основных числовых признаков")
    st.dataframe(data[['price', 'sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 'sqft_living15', 'sqft_lot15']].describe().T.round(3))
    
    st.write("### Корреляционная матрица")
    fig = plt.figure(figsize=(20,10))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Корреляционная матрица')
    st.pyplot(fig)

    data['is_renovated'] = (data['yr_renovated'] > 0).astype('int8')
    data['has_basement'] = (data['sqft_basement'] > 0).astype('int8')
    data = data.drop(columns=['yr_renovated'])
        
    data['price_log'] = np.log(data['price'])
    data['sqft_living_log'] = np.log(data['sqft_living'])
    data['sqft_lot_log'] = np.log(data['sqft_lot'])

    data['sqft_basement_log'] = np.log1p(data['sqft_basement']) 
    data['sqft_living15_log'] = np.log(data['sqft_living15'])
    data['sqft_lot15_log'] = np.log(data['sqft_lot15'])
        
    st.write("### Описательная статистика основных числовых признаков (логарифмированных)")
    st.dataframe(data[['price_log', 'sqft_living_log', 'sqft_lot_log', 'sqft_basement_log', 'sqft_living15_log', 'sqft_lot15_log']].describe().T.round(3))

    st.session_state['eda_data'] = data


elif page == "Визуализация данных":
    st.title("Визуализация зависимостей в наборе данных")
    
    if 'eda_data' not in st.session_state:
        st.warning("Сначала зайдите на страницу 'Информация о наборе данных' для загрузки и предобработки данных.")
    else:
        data = st.session_state['eda_data']
        
        st.subheader("1. Распределение цен")

        fig1 = plt.figure(figsize=(14,10))
        plt.subplot(2, 2, 1)
        sns.histplot(data['price'])
        plt.xlabel('Цена')
        plt.title('Распределение цен')

        plt.subplot(2, 2, 2)
        sns.boxplot(x = data['price'])
        plt.title('Boxplot цен')

        plt.subplot(2, 2, 3)
        sns.histplot(data[data['price'] < 2000000]['price'])
        plt.title('Распределение цен до 2 млн')
        plt.ylabel(' ')
        plt.xlabel('Цена')

        plt.subplot(2, 2, 4)
        sns.histplot(data[data['price'] >= 2000000]['price'])
        plt.title('Распределение цен после 2 млн')
        plt.ylabel(' ')
        plt.xlabel('Цена')
        st.pyplot(fig1)

        st.subheader("2. Распределение логарифмированной цены")

        fig2 = plt.figure(figsize=(15, 6))
        plt.subplot(1, 2, 1)
        sns.histplot(data['price_log'])
        plt.title('Распределение логарифмированной цены')
        plt.xlabel('Цена (log)')

        plt.subplot(1, 2, 2)
        sns.boxplot(x=data['price_log'])
        plt.title('Boxplot логарифмированной цены')
        plt.xlabel('Цена (log)')
        st.pyplot(fig2)

        st.subheader("3. Зависимость цены от площади и участка")

        fig3 = plt.figure(figsize=(20, 12))
        plt.subplot(2, 3, 1)
        sns.histplot(data['sqft_living'], bins=50)
        plt.xlabel('Жилая площадь')
        plt.title('Распределение жилой площади')

        plt.subplot(2, 3, 2)
        sns.scatterplot(data=data, x='sqft_living', y='price', s=10, alpha=0.3)
        plt.title('Зависимость цены от площади')
        plt.xlabel('Жилая площадь')
        plt.ylabel('Цена')

        plt.subplot(2, 3, 3)
        sns.scatterplot(data=data, x='sqft_living', y='price_log', s=10, alpha=0.3)
        plt.title('Зависимость цены (log) от площади')
        plt.xlabel('Жилая площадь')
        plt.ylabel(' ')

        plt.subplot(2, 3, 4)
        sns.histplot(data['sqft_lot'], bins=50)
        plt.xlabel('Земельный участок')
        plt.title('Распределение земельного участка')

        plt.subplot(2, 3, 5)
        sns.scatterplot(data=data, x='sqft_lot', y='price', s=10, alpha=0.3)
        plt.title('Зависимость цены от земельного участка')
        plt.xlabel('Земельный участок')
        plt.ylabel('Цена')

        plt.subplot(2, 3, 6)
        sns.scatterplot(data=data, x='sqft_lot', y='price_log', s=10, alpha=0.3)
        plt.title('Зависимость цены (log) от земельного участка')
        plt.xlabel('Земельный участок')
        plt.ylabel(' ')
        st.pyplot(fig3)

        st.subheader("4. Распределение логарифмированной жилой площади и зависимость ее от цены")

        if 'sqft_living_log' not in data.columns:
            data['sqft_living_log'] = np.log(data['sqft_living'])
        if 'sqft_lot_log' not in data.columns:
            data['sqft_lot_log'] = np.log(data['sqft_lot'])

        fig_log_area = plt.figure(figsize=(18, 12))
        plt.subplot(2, 2, 1)
        sns.histplot(data['sqft_living_log'], bins=50)
        plt.xlabel('Жилая площадь (log)')
        plt.title('Распределение жилой площади (log)')

        plt.subplot(2, 2, 2)
        sns.scatterplot(data=data, x='sqft_living_log', y='price_log', s=10, alpha=0.3)
        plt.title('Зависимость цены (log) от площади (log)')
        plt.xlabel('Жилая площадь (log)')
        plt.ylabel('Цена (log)')

        plt.subplot(2, 2, 3)
        sns.histplot(data['sqft_lot_log'], bins=50) 
        plt.xlabel('Земельный участок (log)')
        plt.title('Распределение земельного участка (log)')

        plt.subplot(2, 2, 4)
        sns.scatterplot(data=data, x='sqft_lot_log', y='price_log', s=10, alpha=0.3)
        plt.title('Зависимость цены (log) от земельного участка (log)')
        plt.xlabel('Земельный участок (log)')
        plt.ylabel('Цена (log)')        
        st.pyplot(fig_log_area)

        st.subheader("5. Динамика цен по месяцам")

        fig_month = plt.figure(figsize=(12,6))
        sns.lineplot(data=data, x='month', y='price', estimator='median', errorbar=None)
        plt.title('Медианная цена по месяцам')
        plt.xlabel('Месяц продажи')
        plt.ylabel('Медианная цена')
        plt.xticks(range(1, 13))
        plt.grid(linestyle='--')
        st.pyplot(fig_month)

        st.subheader("6. Распределение цен по характеристикам дома (спальни, ванны, подвал)")

        fig4 = plt.figure(figsize=(18, 6))
        plt.subplot(1, 3, 1)
        sns.boxplot(data=data, x='bedrooms', y='price_log')
        plt.xlabel('Количество спален')
        plt.ylabel('Цена (log)')

        plt.subplot(1, 3, 2)
        sns.boxplot(data=data, x='bathrooms', y='price_log')
        plt.xlabel('Количество ванн')
        plt.ylabel(' ')

        plt.subplot(1, 3, 3)
        sns.boxplot(data=data, x='has_basement', y='price_log')
        plt.xticks([0, 1], ['Нет подвала', 'Есть подвал'])
        plt.xlabel(' ')
        plt.ylabel(' ')
        st.pyplot(fig4)

        st.subheader("7. Влияние возраста и реконструкции")

        fig5 = plt.figure(figsize=(18, 6))
        plt.subplot(1, 3, 1)
        sns.histplot(data['house_age'], bins=50)
        plt.title('Распределение возраста домов')
        plt.xlabel('Возраст дома')
        plt.ylabel('Количество')

        plt.subplot(1, 3, 2)
        sns.scatterplot(data=data, x='house_age', y='price_log', s=15, alpha=0.3)
        plt.title('Зависимость цены (log) от возраста')
        plt.xlabel('Возраст дома')
        plt.ylabel('Цена (log)')

        plt.subplot(1, 3, 3)
        sns.boxplot(data=data, x='is_renovated', y='price_log')
        plt.title('Сравнение цен реконструированных домов')
        plt.xlabel('Наличие реконструции')
        plt.ylabel(' ')
        plt.xticks([0, 1], ['Нет', 'Да'])
        st.pyplot(fig5)

        st.subheader("8. География, вид и соседи")

        fig6 = plt.figure(figsize=(20, 6))
        plt.subplot(1,3,1)
        sc = plt.scatter(data['long'], data['lat'], c = data['price_log'], s=10, alpha=0.4)
        plt.colorbar(sc, label= 'Цена (log)')
        plt.xlabel('Долгота')
        plt.ylabel('Широта')
        plt.title('Зависимость цены от географического положения')

        plt.subplot(1,3,2)
        sns.boxplot(data=data, x='waterfront', y='price_log')
        plt.xticks([0, 1], ['Без вида на море', 'С видом на море'])
        plt.title('Сравнение цен домов с видом на море и без')
        plt.ylabel(' ')
        plt.xlabel(' ')

        plt.subplot(1, 3, 3)
        data['sqft_living15_log'] = np.log(data['sqft_living15'])
        sns.scatterplot(data=data, x='sqft_living15_log', y='price_log', alpha=0.3, s=15)
        plt.title('Зависимость цены от соседей')
        plt.xlabel('Средняя площадь 15 соседей (log)')
        plt.ylabel(' ')
        st.pyplot(fig6)

        st.subheader("9. Влияние эстетики и качества (Вид, Состояние, Оценка)")

        fig7 = plt.figure(figsize=(20, 6))
        plt.subplot(1, 4, 1)
        sns.boxplot(data=data, x='view', y='price_log')
        plt.title('Зависимость цены от вида')
        plt.xlabel('Вид (0 - плохой, 4 - отличный)')
        plt.ylabel('Цена (log)')

        plt.subplot(1, 4, 2)
        sns.boxplot(data=data, x='condition', y='price_log')
        plt.title('Зависимость цены от состояния')
        plt.xlabel('Состояние (1 - плохое, 5 - отличное)')
        plt.ylabel(' ')

        plt.subplot(1, 4, 3)
        sns.boxplot(data=data, x='floors', y='price_log')
        plt.title('Зависимость цены от кол-ва этажей')
        plt.xlabel('Количество этажей')
        plt.ylabel(' ')

        plt.subplot(1, 4, 4)
        sns.boxplot(data=data, x='grade', y='price_log')
        plt.title('Зависимость цены от оценок')
        plt.xlabel('Оценка качества')
        plt.ylabel(' ')
        st.pyplot(fig7)

elif page == "Предсказание моделей ML":
    st.title("Получение предсказания")
    
    @st.cache_data
    def load_training_meta():
        df_train = pd.read_csv('data/house_regression.csv') 
        X_train = df_train.drop(columns=['price_log'])
        scaler = StandardScaler()
        scaler.fit(X_train)
        return scaler, X_train.columns.tolist()

    scaler, feature_names = load_training_meta()

    model_options = {
        "ML1: Ridge (Polynomial)": ("models/poly_ridge_model.pkl", "pipeline"),
        "ML2: Gradient Boosting": ("models/gb_model.pkl", "raw"),
        "ML3: LightGBM": ("models/lgb_model.txt", "lgb"),
        "ML4: Bagging": ("models/bag_model.pkl", "raw"),
        "ML5: Stacking": ("models/stack_model.pkl", "raw"),
        "ML6: Neural Network (KerasTuner)": ("models/ml6_model.h5", "keras")
    }
    
    model_choice = st.selectbox("Выберите модель ML:", list(model_options.keys()))
    
    st.write("---")
    
    input_method = st.radio("Способ ввода данных:", ["Ручной ввод", "Загрузка CSV файла"])
    
    input_data = None 
    run_prediction = False 
    
    if input_method == "Загрузка CSV файла":
        st.write("Загрузите оригинальный датасет")
        uploaded_file = st.file_uploader("Выберите CSV файл", type=['csv'])
        
        if uploaded_file is not None:
            input_data = pd.read_csv(uploaded_file)
            st.success("Файл успешно загружен!")
            st.dataframe(input_data)
            
            with st.spinner("Автоматическая предобработка данных..."):
                try:
                    if 'date' in input_data.columns and 'yr_built' in input_data.columns:
                        input_data['date'] = pd.to_datetime(input_data['date'], errors='coerce')
                        input_data['year'] = input_data['date'].dt.year
                        input_data['month'] = input_data['date'].dt.month
                        input_data['house_age'] = input_data['year'] - input_data['yr_built']
                    
                    if 'yr_renovated' in input_data.columns and 'is_renovated' not in input_data.columns:
                        input_data['is_renovated'] = (input_data['yr_renovated'] > 0).astype('int8')
                    if 'sqft_basement' in input_data.columns and 'has_basement' not in input_data.columns:
                        input_data['has_basement'] = (input_data['sqft_basement'] > 0).astype('int8')

                    if 'sqft_living' in input_data.columns and 'sqft_living_log' not in input_data.columns:
                        input_data['sqft_living_log'] = np.log(input_data['sqft_living'].replace(0, 1))
                    if 'sqft_lot' in input_data.columns and 'sqft_lot_log' not in input_data.columns:
                        input_data['sqft_lot_log'] = np.log(input_data['sqft_lot'].replace(0, 1))
                    if 'sqft_basement' in input_data.columns and 'sqft_basement_log' not in input_data.columns:
                        input_data['sqft_basement_log'] = np.log1p(input_data['sqft_basement'])
                    if 'sqft_living15' in input_data.columns and 'sqft_living15_log' not in input_data.columns:
                        input_data['sqft_living15_log'] = np.log(input_data['sqft_living15'].replace(0, 1))
                    if 'sqft_lot15' in input_data.columns and 'sqft_lot15_log' not in input_data.columns:
                        input_data['sqft_lot15_log'] = np.log(input_data['sqft_lot15'].replace(0, 1))

                    missing_cols = set(feature_names) - set(input_data.columns)
                    
                    if missing_cols:
                        st.error(f"Ошибка валидации! Даже после автоматической предобработки не хватает столбцов:\n{list(missing_cols)}")
                        input_data = None
                    else:
                        st.info("Данные успешно прошли автоматическую предобработку и готовы к предсказанию!")
                        run_prediction = st.button("Предсказать по загруженному файлу")
                        
                except Exception as e:
                    st.error(f"Произошла ошибка при обработке файла: {e}")
                    input_data = None
                
    else:
        st.write("Введите реальные значения признаков вручную:")
        with st.form("manual_input_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                bedrooms = st.number_input("Количество спален", min_value=0, max_value=33, value=2)
                bathrooms = st.number_input("Количество ванн", min_value=0.0, max_value=10.0, value=1.0)
                floors = st.number_input("Количество этажей", min_value=1.0, max_value=4.0, value=1.0)
                waterfront = st.selectbox("Вид на воду (0 - нет, 1 - есть)", [0, 1])
                view = st.selectbox("Оценка вида (0-4)", [0, 1, 2, 3, 4])
                condition = st.selectbox("Состояние (1-5)", [1, 2, 3, 4, 5])
            
            with col2:
                grade = st.number_input("Оценка качества (1-13)", min_value=1, max_value=13, value=7)
                month = st.number_input("Месяц продажи", min_value=1, max_value=12, value=5)
                house_age = st.number_input("Возраст дома (лет)", min_value=0, max_value=150, value=20)
                is_renovated = st.selectbox("Был реконструирован (0/1)", [0, 1])
                has_basement = st.selectbox("Есть подвал (0/1)", [0, 1])
                lat = st.number_input("Широта (lat)", value=47.56)
                
            with col3:
                long = st.number_input("Долгота (long)", value=-122.21)
                sqft_living = st.number_input("Жилая площадь (кв. фут)", min_value=1, value=1800)
                sqft_lot = st.number_input("Площадь участка (кв. фут)", min_value=1, value=5000)
                sqft_basement = st.number_input("Площадь подвала (кв. фут)", min_value=0, value=0)
                sqft_living15 = st.number_input("Жилая пл. соседей (кв. фут)", min_value=1, value=1800)
                sqft_lot15 = st.number_input("Пл. участка соседей (кв. фут)", min_value=1, value=5000)
            
            run_prediction = st.form_submit_button(label="Предсказать цену")
            
        if run_prediction:
            sqft_living_log = np.log(sqft_living)
            sqft_lot_log = np.log(sqft_lot)
            sqft_basement_log = np.log1p(sqft_basement)
            sqft_living15_log = np.log(sqft_living15)
            sqft_lot15_log = np.log(sqft_lot15)

            input_dict = {
                'bedrooms': [bedrooms], 'bathrooms': [bathrooms], 'floors': [floors],
                'waterfront': [waterfront], 'view': [view], 'condition': [condition],
                'grade': [grade], 'lat': [lat], 'long': [long], 'month': [month],
                'house_age': [house_age], 'is_renovated': [is_renovated], 'has_basement': [has_basement],
                'sqft_living_log': [sqft_living_log], 'sqft_lot_log': [sqft_lot_log],
                'sqft_basement_log': [sqft_basement_log], 'sqft_living15_log': [sqft_living15_log],
                'sqft_lot15_log': [sqft_lot15_log]
            }
            input_data = pd.DataFrame(input_dict)

    if run_prediction and input_data is not None:
        with st.spinner('Загрузка модели и подготовка предсказания...'):
            try:
                input_data = input_data[feature_names]
                file_name, model_type = model_options[model_choice]
            
                if model_type == "keras":
                    data_to_predict = scaler.transform(input_data)
                else:
                    data_to_predict = input_data

                if model_type in ["pipeline", "raw"]:
                    model = joblib.load(file_name)
                    pred_log = model.predict(data_to_predict)
                
                elif model_type == "lgb":
                    model = lgb.Booster(model_file=file_name)
                    pred_log = model.predict(data_to_predict)
                    
                elif model_type == "keras":
                    model = load_model(file_name, compile=False)
                    pred_log = model.predict(data_to_predict).flatten()
                    
                if len(pred_log) == 1:
                    val_log = pred_log[0]
                    val_exp = np.exp(val_log)
                    
                    st.info(f"Ожидаемая стоимость недвижимости: **${val_exp:,.2f}**")
                else:
                    res_df = input_data.copy()
                    res_df['Предсказанная цена ($)'] = np.exp(pred_log).round(2)
                    st.success("Все предсказания успешно добавлены!")
                    st.dataframe(res_df[['Предсказанная цена ($)']])
                    
            except FileNotFoundError:
                st.error(f"Файл модели '{file_name}' не найден в рабочей директории приложения.")
            except Exception as e:
                st.error(f"Произошла непредвиденная ошибка при вычислении: {e}")
