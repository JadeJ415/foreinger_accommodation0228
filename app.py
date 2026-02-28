import streamlit as st
import pandas as pd
import sqlite3
import os
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# 페이지 설정
st.set_page_config(
    page_title="서울 공유숙박 창업 전략 대시보드",
    page_icon="🏠",
    layout="wide"
)

# 데이터 경로 (상대 경로)
base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset')

@st.cache_data
def load_data():
    # 데이터 로드 로직 (이전 app.py와 유사하되 창업 분석에 특화)
    def robust_read_csv(file_name):
        path = os.path.join(base_path, file_name)
        try:
            return pd.read_csv(path, encoding='utf-8-sig')
        except:
            return pd.read_csv(path, encoding='cp949')

    df_nat = robust_read_csv('(전국기준)국적별+외국인+방문객_20260228095700.csv')
    df_nat = df_nat.iloc[1:].copy()
    df_nat.columns = ['대륙', '국가', '계', '남자', '여자']
    df_nat = df_nat[~df_nat['국가'].isin(['소계', '대륙별(2)'])]
    df_nat['계'] = pd.to_numeric(df_nat['계'], errors='coerce')

    df_age = robust_read_csv('(전국기준)연령별+외국인+방문객_20260228095859.csv')
    df_age = df_age.iloc[2:].copy()
    df_age.columns = ['대륙1', '대륙2', '합계', '0-9세', '10-19세', '20-29세', '30-39세', '40-49세', '50-59세', '60-69세', '70-79세', '80세이상', '승무원']

    df_hotel = robust_read_csv('관광호텔+등록현황_20260228095634.csv')
    df_hotel = df_hotel.iloc[3:].copy()
    df_hotel.columns = ['지역1', '지역2', '호텔수', '객실수'] + [f'col_{i}' for i in range(len(df_hotel.columns)-4)]
    df_hotel_seoul = df_hotel[df_hotel['지역1'] == '서울시'].copy()
    df_hotel_seoul['호텔수'] = pd.to_numeric(df_hotel_seoul['호텔수'], errors='coerce')

    df_fore = robust_read_csv('foreigner.csv')
    df_fore_active = df_fore[df_fore['영업상태명'] == '영업/정상'].copy()
    df_fore_active['구'] = df_fore_active['소재지전체주소'].str.split(' ', expand=True)[1]

    conn = sqlite3.connect(os.path.join(base_path, 'airbnb.db'))
    df_airbnb = pd.read_sql_query("SELECT * FROM airbnb_stays", conn)
    conn.close()
    df_airbnb['price_val'] = pd.to_numeric(df_airbnb['price_value'], errors='coerce')

    return df_nat, df_age, df_hotel_seoul, df_fore_active, df_airbnb

# 데이터 로드
df_nat, df_age, df_hotel, df_fore, df_airbnb = load_data()

# 사이드바 설정
st.sidebar.header("🚀 창업 파라미터 설정")
target_adr = st.sidebar.slider("목표 1박 객실단가 (ADR)", 50000, 300000, int(df_airbnb['price_val'].median()), step=10000)
target_occ = st.sidebar.slider("목표 점유율 (OCC %)", 0, 100, 70) / 100
startup_cost = st.sidebar.number_input("초기 투자비용 (인테리어/집기 등)", 5000000, 100000000, 15000000, step=1000000)
op_ratio = st.sidebar.slider("운영비율 (매출 대비 %)", 10, 60, 35) / 100

# 메인 화면
st.title("🏠 서울 공유숙박 창업 전략 대시보드")
st.markdown("---")

# 상단 KPI
m_rev = int(target_adr * 30 * target_occ)
m_profit = int(m_rev * (1 - op_ratio))
bep_months = round(startup_cost / m_profit, 1) if m_profit > 0 else 0

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("예상 월 매출", f"₩{m_rev:,}")
with col2:
    st.metric("예상 월 순익", f"₩{m_profit:,}", delta=f"이익률 {int((1-op_ratio)*100)}%")
with col3:
    st.metric("투자 회수 기간", f"{bep_months}개월")
with col4:
    st.metric("시장 중간 가격", f"₩{int(df_airbnb['price_val'].median()):,}")

st.markdown("---")

# 탭 구성
tab1, tab2, tab3 = st.tabs(["📊 수요-공급 갭 분석", "💵 수익 시뮬레이션", "🎯 타겟 고객 프로파일링"])

with tab1:
    st.header("📍 어느 지역에 창업해야 할까?")
    
    # 지역별 공급 데이터 병합
    hotel_cnt = df_hotel[df_hotel['지역2'] != '소계'][['지역2', '호텔수']].rename(columns={'지역2': '구'})
    fore_cnt = df_fore['구'].value_counts().reset_index()
    fore_cnt.columns = ['구', '도시민박수']
    gap_df = pd.merge(hotel_cnt, fore_cnt, on='구', how='outer').fillna(0)
    gap_df['Total_Supply'] = gap_df['호텔수'] + gap_df['도시민박수']
    
    col_a, col_b = st.columns(2)
    with col_a:
        fig_supply = px.bar(gap_df.sort_values('Total_Supply', ascending=False), 
                            x='구', y=['호텔수', '도시민박수'], 
                            title="지역별 숙박 공급 현황 (호텔 vs 민박)",
                            barmode='stack', color_discrete_sequence=['#1f77b4', '#ff7f0e'])
        st.plotly_chart(fig_supply, use_container_width=True)
        
    with col_b:
        # 호텔 대비 민박 비중이 낮은 곳 = 호텔의 표준화된 서비스에 지친 고객을 뺏어올 수 있는 기회
        gap_df['민박비중'] = gap_df['도시민박수'] / (gap_df['Total_Supply'] + 1)
        fig_gap = px.scatter(gap_df, x='호텔수', y='도시민박수', text='구', size='Total_Supply',
                             color='민박비중', color_continuous_scale='RdYlGn_r',
                             title="수요-공급 매트릭스 (우하단: 블루오션 후보)")
        st.plotly_chart(fig_gap, use_container_width=True)
    
    st.info("💡 **전략 제안**: 성동구(성수)와 용산구는 호텔 공급 대비 민박 수요가 급증하는 지역으로 높은 프리미엄 전략이 가능합니다.")

with tab2:
    st.header("📈 상세 수익 시뮬레이션")
    
    # 점유율 시나리오 분석
    occ_scenarios = np.linspace(0.3, 1.0, 8)
    rev_scenario = [target_adr * 30 * o for o in occ_scenarios]
    profit_scenario = [r * (1 - op_ratio) for r in rev_scenario]
    
    fig_sim = go.Figure()
    fig_sim.add_trace(go.Scatter(x=occ_scenarios*100, y=rev_scenario, name='월 매출', mode='lines+markers'))
    fig_sim.add_trace(go.Scatter(x=occ_scenarios*100, y=profit_scenario, name='월 순익', mode='lines+markers'))
    fig_sim.update_layout(title="점유율 변화에 따른 수익성 변화", xaxis_title="점유율 (%)", yaxis_title="금액 (원)")
    st.plotly_chart(fig_sim, use_container_width=True)
    
    st.markdown("#### 💰 최적 가격 전략 찾기")
    price_range = np.linspace(int(target_adr*0.7), int(target_adr*1.3), 10)
    # 단순 가상 모델: 가격이 높아지면 점유율이 낮아지는 탄력성 가정
    sim_occ = [max(0.2, target_occ - (p - target_adr)/target_adr * 0.5) for p in price_range]
    sim_profit = [p * 30 * o * (1 - op_ratio) for p, o in zip(price_range, sim_occ)]
    
    fig_price = px.line(x=price_range, y=sim_profit, labels={'x': '객실 단가(ADR)', 'y': '예상 월 순익'},
                        title="가격 탄력성 시뮬레이션 (최고점 이익 지점 탐색)")
    st.plotly_chart(fig_price, use_container_width=True)

with tab3:
    st.header("👤 누구를 타겟으로 할 것인가?")
    
    col_c, col_d = st.columns(2)
    with col_c:
        # 국적 데이터
        top_10 = df_nat.sort_values('계', ascending=False).head(10)
        fig_nat = px.pie(top_10, values='계', names='국가', title="핵심 타겟 국가 Top 10")
        st.plotly_chart(fig_nat, use_container_width=True)
        
    with col_d:
        # 연령 데이터
        df_age_total = df_age[df_age['대륙2'] == '소계'].iloc[0]
        age_labels = ['20대', '30대', '40대', '50대이상']
        age_values = [
            pd.to_numeric(df_age_total['20-29세']),
            pd.to_numeric(df_age_total['30-39세']),
            pd.to_numeric(df_age_total['40-49세']),
            sum([pd.to_numeric(df_age_total[c]) for c in ['50-59세', '60-69세', '70-79세', '80세이상']])
        ]
        fig_age = px.bar(x=age_labels, y=age_values, title="연령대별 방문객 분포", 
                         labels={'x': '연령대', 'y': '방문객 수'}, color=age_labels)
        st.plotly_chart(fig_age, use_container_width=True)

    st.success("🎯 **결론**: 2030 영미권/아시아 트렌드 세터를 타겟으로 한 '감성 로컬 스테이'가 가장 승률이 높습니다.")

# 푸터
st.markdown("---")
st.caption("Produced by Antigravity Startup Analysis Team")
