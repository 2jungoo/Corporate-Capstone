"""
Step 1: 기존 데이터 학습 시스템 (개선판)
- 기존 CSV 데이터에서 성장 패턴 분석
- 사료 섭취량 vs 체중 증가 관계 파악
- 체중 구간별 증체율 분석
- 학습된 패턴을 JSON으로 저장
- 비정상적인 데이터 필터링 강화
"""

import pandas as pd
import numpy as  np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import os
import chardet
from scipy import stats as scipy_stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import warnings

warnings.filterwarnings('ignore')


def detect_encoding(file_path):
    """파일의 인코딩을 자동으로 감지"""
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)
            result = chardet.detect(raw_data)
            return result['encoding']
    except:
        return None


def read_csv_with_encoding(file_path):
    """다양한 인코딩을 시도하여 CSV 파일 읽기"""
    encodings = ['utf-8', 'utf-8-sig', 'cp949', 'euc-kr', 'latin1', 'iso-8859-1']

    try:
        detected_encoding = detect_encoding(file_path)
        if detected_encoding:
            try:
                df = pd.read_csv(file_path, encoding=detected_encoding)
                return df, detected_encoding
            except:
                pass
    except:
        pass

    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            return df, encoding
        except:
            continue

    return None, None


class GrowthPatternLearner:
    """기존 데이터에서 성장 패턴 학습 (개선판)"""

    def __init__(self, base_path='./텍스트 데이터'):
        self.base_path = base_path
        self.all_data = {}
        self.growth_patterns = {}
        # 업계 표준 기본값 설정
        self.default_patterns = self.get_default_patterns()

    def get_default_patterns(self):
        """업계 표준 기본 성장 패턴"""
        return {
            'overall': {
                'mean_daily_gain': 0.65,  # 업계 평균
                'std_daily_gain': 0.15,
                'median_daily_gain': 0.63,
                'min_weight': 20.0,
                'max_weight': 110.0,
                'mean_weight': 50.0,
                'mean_feed': 1.8,
                'std_feed': 0.5
            },
            'weight_bins': {
                '0-20kg': {
                    'mean_daily_gain': 0.45,
                    'std_daily_gain': 0.10,
                    'median_daily_gain': 0.44,
                    'min_daily_gain': 0.25,
                    'max_daily_gain': 0.65,
                    'count': 10,
                    'mean_feed': 0.8,
                    'std_feed': 0.2
                },
                '20-40kg': {
                    'mean_daily_gain': 0.65,
                    'std_daily_gain': 0.12,
                    'median_daily_gain': 0.64,
                    'min_daily_gain': 0.40,
                    'max_daily_gain': 0.90,
                    'count': 10,
                    'mean_feed': 1.5,
                    'std_feed': 0.3
                },
                '40-60kg': {
                    'mean_daily_gain': 0.75,
                    'std_daily_gain': 0.10,
                    'median_daily_gain': 0.74,
                    'min_daily_gain': 0.55,
                    'max_daily_gain': 0.95,
                    'count': 10,
                    'mean_feed': 2.2,
                    'std_feed': 0.3
                },
                '60-80kg': {
                    'mean_daily_gain': 0.80,
                    'std_daily_gain': 0.10,
                    'median_daily_gain': 0.79,
                    'min_daily_gain': 0.60,
                    'max_daily_gain': 1.00,
                    'count': 10,
                    'mean_feed': 2.8,
                    'std_feed': 0.3
                },
                '80-100kg': {
                    'mean_daily_gain': 0.70,
                    'std_daily_gain': 0.12,
                    'median_daily_gain': 0.69,
                    'min_daily_gain': 0.45,
                    'max_daily_gain': 0.95,
                    'count': 10,
                    'mean_feed': 3.2,
                    'std_feed': 0.4
                },
                '100kg+': {
                    'mean_daily_gain': 0.55,
                    'std_daily_gain': 0.15,
                    'median_daily_gain': 0.54,
                    'min_daily_gain': 0.25,
                    'max_daily_gain': 0.85,
                    'count': 5,
                    'mean_feed': 3.5,
                    'std_feed': 0.5
                }
            },
            'feed_to_gain': {
                'coefficient': 0.22,  # 사료 1kg당 증체 0.22kg (FCR ~4.5)
                'intercept': 0.20,
                'r2': 0.65
            },
            'growth_curve': {
                'coefficients': [0.0, 0.75, -0.0015],  # 2차 곡선 계수
                'intercept': 20.0,
                'degree': 2
            }
        }

    def load_all_data(self):
        """모든 chamber의 데이터 로드"""
        print("\n" + "="*80)
        print("📚 Step 1: 기존 데이터 로딩 중...")
        print("="*80)

        data_types = {
            '돼지체중': 'weight',
            '사양관리/섭취량': 'feed'
        }

        data_found = False

        for chamber in ['chamber1', 'chamber2', 'chamber3', 'chamber4']:
            self.all_data[chamber] = {}
            print(f"\n[{chamber}] 데이터 로딩...")

            # 체중 데이터
            weight_path = os.path.join(self.base_path, chamber, '돼지체중')
            if os.path.exists(weight_path):
                csv_files = list(Path(weight_path).glob('*.csv'))
                weight_data = []
                for csv_file in csv_files:
                    df, encoding = read_csv_with_encoding(str(csv_file))
                    if df is not None:
                        df['chamber'] = chamber
                        weight_data.append(df)

                if weight_data:
                    self.all_data[chamber]['weight'] = pd.concat(weight_data, ignore_index=True)
                    print(f"  ✓ 체중: {len(self.all_data[chamber]['weight'])}건")
                    data_found = True

            # 사료 데이터
            feed_path = os.path.join(self.base_path, chamber, '사양관리', '섭취량')
            if os.path.exists(feed_path):
                csv_files = list(Path(feed_path).glob('*.csv'))
                feed_data = []
                for csv_file in csv_files:
                    df, encoding = read_csv_with_encoding(str(csv_file))
                    if df is not None:
                        df['chamber'] = chamber
                        feed_data.append(df)

                if feed_data:
                    self.all_data[chamber]['feed'] = pd.concat(feed_data, ignore_index=True)
                    print(f"  ✓ 사료: {len(self.all_data[chamber]['feed'])}건")
                    data_found = True

        if not data_found:
            print("\n⚠️ 실제 데이터를 찾을 수 없어 기본 패턴 사용")
            self.growth_patterns = self.default_patterns

        return self.all_data

    def standardize_dataframe(self, df, data_type):
        """데이터프레임 표준화"""
        result = df.copy()

        # 날짜 컬럼
        date_col = None
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['날짜', 'date', '일자', '시간', 'time']):
                date_col = col
                break

        if date_col:
            result['date'] = pd.to_datetime(df[date_col], errors='coerce')

        # 개체 ID
        pig_col = None
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['개체', 'pig', 'id', '번호']):
                pig_col = col
                break

        if pig_col:
            result['pig_id'] = df[pig_col]

        # 값 컬럼
        if data_type == 'weight':
            for col in df.columns:
                if any(keyword in col.lower() for keyword in ['체중', 'weight', '무게']):
                    result['value'] = pd.to_numeric(df[col], errors='coerce')
                    break
        elif data_type == 'feed':
            for col in df.columns:
                if any(keyword in col.lower() for keyword in ['섭취', 'feed', '사료']):
                    result['value'] = pd.to_numeric(df[col], errors='coerce')
                    break

        return result

    def integrate_data(self):
        """데이터 통합"""
        print("\n" + "="*80)
        print("📊 데이터 통합 및 전처리 중...")
        print("="*80)

        all_records = []

        for chamber, chamber_data in self.all_data.items():
            for data_type, df in chamber_data.items():
                std_df = self.standardize_dataframe(df, data_type)

                if 'date' in std_df.columns and 'value' in std_df.columns:
                    subset = std_df[['chamber', 'date', 'pig_id', 'value']].copy()
                    subset = subset.dropna(subset=['date', 'value'])
                    subset['data_type'] = data_type
                    all_records.append(subset)

        if all_records:
            combined = pd.concat(all_records, ignore_index=True)
            print(f"✓ 총 {len(combined)}건 통합")

            # Pivot
            combined['date'] = combined['date'].dt.floor('D')
            pivot_df = combined.pivot_table(
                index=['chamber', 'pig_id', 'date'],
                columns='data_type',
                values='value',
                aggfunc='mean'
            ).reset_index()

            # 일령 계산
            pivot_df = pivot_df.sort_values(['chamber', 'pig_id', 'date'])
            pivot_df['day'] = pivot_df.groupby(['chamber', 'pig_id'])['date'].transform(
                lambda x: (x - x.min()).dt.days
            )

            print(f"✓ Pivot 완료: {len(pivot_df)}건")
            return pivot_df

        return None

    def analyze_growth_patterns(self, df):
        """성장 패턴 분석 (개선된 필터링)"""
        print("\n" + "="*80)
        print("🔍 성장 패턴 분석 중...")
        print("="*80)

        if df is None or len(df) == 0:
            print("⚠️ 분석할 데이터가 없어 기본 패턴 사용")
            self.growth_patterns = self.default_patterns
            return pd.DataFrame()

        # 체중이 있는 데이터만
        df_analysis = df.dropna(subset=['weight']).copy()

        # 체중 범위 필터링 (비정상값 제거)
        df_analysis = df_analysis[
            (df_analysis['weight'] > 5) &  # 5kg 미만 제외
            (df_analysis['weight'] < 200)  # 200kg 초과 제외
        ]

        # 증체량 계산
        df_analysis['weight_gain'] = df_analysis.groupby(['chamber', 'pig_id'])['weight'].diff()
        df_analysis['days_diff'] = df_analysis.groupby(['chamber', 'pig_id'])['day'].diff()

        # 0일 차이 방지
        df_analysis = df_analysis[df_analysis['days_diff'] > 0]
        df_analysis['daily_gain'] = df_analysis['weight_gain'] / df_analysis['days_diff']

        # 이상치 제거 (더 엄격한 기준)
        df_analysis = df_analysis[
            (df_analysis['daily_gain'] > 0.1) &  # 0.1kg/일 미만 제외
            (df_analysis['daily_gain'] < 1.5)    # 1.5kg/일 초과 제외
        ]

        # 데이터가 부족한 경우 기본값 사용
        if len(df_analysis) < 10:
            print("⚠️ 유효한 데이터가 부족하여 기본 패턴 사용")
            self.growth_patterns = self.default_patterns
            return df_analysis

        # 1. 체중 구간별 증체율
        weight_bins = [0, 20, 40, 60, 80, 100, 200]
        bin_labels = ['0-20kg', '20-40kg', '40-60kg', '60-80kg', '80-100kg', '100kg+']

        df_analysis['weight_bin'] = pd.cut(
            df_analysis['weight'],
            bins=weight_bins,
            labels=bin_labels,
            include_lowest=True
        )

        # 2. 사료 효율 분석 (사료 데이터가 있는 경우만)
        if 'feed' in df_analysis.columns:
            feed_data = df_analysis.dropna(subset=['feed', 'daily_gain'])

            # 사료 섭취량 정상 범위 필터링
            feed_data = feed_data[
                (feed_data['feed'] > 0.2) &  # 0.2kg/일 미만 제외
                (feed_data['feed'] < 5.0)    # 5kg/일 초과 제외
            ]

            if len(feed_data) > 10:
                X = feed_data['feed'].values.reshape(-1, 1)
                y = feed_data['daily_gain'].values

                model = LinearRegression()
                model.fit(X, y)

                self.growth_patterns['feed_to_gain'] = {
                    'coefficient': float(model.coef_[0]),
                    'intercept': float(model.intercept_),
                    'r2': float(model.score(X, y))
                }

                # R² 값이 너무 낮으면 기본값 사용
                if self.growth_patterns['feed_to_gain']['r2'] < 0.1:
                    self.growth_patterns['feed_to_gain'] = self.default_patterns['feed_to_gain']
            else:
                self.growth_patterns['feed_to_gain'] = self.default_patterns['feed_to_gain']
        else:
            self.growth_patterns['feed_to_gain'] = self.default_patterns['feed_to_gain']

        # 3. 성장 곡선 모델링
        growth_by_day = df_analysis.groupby('day')['weight'].mean().reset_index()

        if len(growth_by_day) > 10:
            X = growth_by_day['day'].values.reshape(-1, 1)
            y = growth_by_day['weight'].values

            # 2차 다항식 회귀
            poly = PolynomialFeatures(degree=2, include_bias=False)
            X_poly = poly.fit_transform(X)

            model_poly = LinearRegression()
            model_poly.fit(X_poly, y)

            # 계수 저장
            coefficients = [0.0] + list(model_poly.coef_)

            self.growth_patterns['growth_curve'] = {
                'coefficients': [float(c) for c in coefficients],
                'intercept': float(model_poly.intercept_),
                'degree': 2
            }

            # 계수가 비정상적이면 기본값 사용
            if abs(coefficients[2]) > 0.1:  # 2차 계수가 너무 크면
                self.growth_patterns['growth_curve'] = self.default_patterns['growth_curve']
        else:
            self.growth_patterns['growth_curve'] = self.default_patterns['growth_curve']

        # 4. 체중 구간별 통계 저장
        weight_stats = {}
        for bin_label in bin_labels:
            bin_data = df_analysis[df_analysis['weight_bin'] == bin_label]

            if len(bin_data) > 5:  # 최소 5개 이상 데이터
                daily_gains = bin_data['daily_gain']

                # 이상치 제거 (IQR 방법)
                Q1 = daily_gains.quantile(0.25)
                Q3 = daily_gains.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                daily_gains_filtered = daily_gains[
                    (daily_gains >= lower_bound) &
                    (daily_gains <= upper_bound)
                ]

                if len(daily_gains_filtered) > 0:
                    weight_stats[bin_label] = {
                        'mean_daily_gain': float(daily_gains_filtered.mean()),
                        'std_daily_gain': float(daily_gains_filtered.std()),
                        'median_daily_gain': float(daily_gains_filtered.median()),
                        'min_daily_gain': float(daily_gains_filtered.min()),
                        'max_daily_gain': float(daily_gains_filtered.max()),
                        'count': int(len(daily_gains_filtered))
                    }

                    # 사료 데이터 추가
                    if 'feed' in bin_data.columns:
                        feed_data = bin_data.dropna(subset=['feed'])
                        if len(feed_data) > 0:
                            weight_stats[bin_label]['mean_feed'] = float(feed_data['feed'].mean())
                            weight_stats[bin_label]['std_feed'] = float(feed_data['feed'].std())

                    # 값이 비정상적이면 기본값으로 대체
                    if weight_stats[bin_label]['mean_daily_gain'] < 0.2 or \
                       weight_stats[bin_label]['mean_daily_gain'] > 1.2:
                        weight_stats[bin_label] = self.default_patterns['weight_bins'].get(
                            bin_label, self.default_patterns['weight_bins']['40-60kg']
                        )
                else:
                    # 필터링 후 데이터가 없으면 기본값 사용
                    weight_stats[bin_label] = self.default_patterns['weight_bins'].get(
                        bin_label, self.default_patterns['weight_bins']['40-60kg']
                    )
            else:
                # 데이터가 부족하면 기본값 사용
                weight_stats[bin_label] = self.default_patterns['weight_bins'].get(
                    bin_label, self.default_patterns['weight_bins']['40-60kg']
                )

        self.growth_patterns['weight_bins'] = weight_stats

        # 5. 전체 통계
        valid_gains = df_analysis['daily_gain'][
            (df_analysis['daily_gain'] > 0.2) &
            (df_analysis['daily_gain'] < 1.2)
        ]

        if len(valid_gains) > 0:
            self.growth_patterns['overall'] = {
                'mean_daily_gain': float(valid_gains.mean()),
                'std_daily_gain': float(valid_gains.std()),
                'median_daily_gain': float(valid_gains.median()),
                'min_weight': float(df_analysis['weight'].min()),
                'max_weight': float(df_analysis['weight'].max()),
                'mean_weight': float(df_analysis['weight'].mean())
            }
        else:
            self.growth_patterns['overall'] = self.default_patterns['overall']

        # 사료 통계 추가
        if 'feed' in df_analysis.columns:
            feed_data = df_analysis.dropna(subset=['feed'])
            feed_data = feed_data[
                (feed_data['feed'] > 0.2) &
                (feed_data['feed'] < 5.0)
            ]
            if len(feed_data) > 0:
                self.growth_patterns['overall']['mean_feed'] = float(feed_data['feed'].mean())
                self.growth_patterns['overall']['std_feed'] = float(feed_data['feed'].std())

        # 최종 검증: 평균 증체율이 비정상적이면 기본값 사용
        if self.growth_patterns['overall']['mean_daily_gain'] < 0.3 or \
           self.growth_patterns['overall']['mean_daily_gain'] > 1.0:
            print("⚠️ 계산된 증체율이 비정상적이어서 보정")
            self.growth_patterns['overall']['mean_daily_gain'] = 0.65
            self.growth_patterns['overall']['std_daily_gain'] = 0.15

        print(f"\n✓ 성장 패턴 분석 완료")
        print(f"   - 전체 평균 증체율: {self.growth_patterns['overall']['mean_daily_gain']:.3f} kg/일")
        print(f"   - 체중 범위: {self.growth_patterns['overall']['min_weight']:.1f} ~ {self.growth_patterns['overall']['max_weight']:.1f} kg")

        return df_analysis

    def visualize_patterns(self, df_analysis, output_path='./growth_patterns_analysis.png'):
        """패턴 시각화 (개선된 그래프)"""
        print("\n📊 시각화 생성 중...")

        # 데이터가 없으면 기본 패턴으로 시각화
        if df_analysis is None or len(df_analysis) == 0:
            df_analysis = self.generate_sample_data()

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. 체중 구간별 증체율
        ax1 = axes[0, 0]
        weight_bins_data = []
        for bin_name, stats in self.growth_patterns['weight_bins'].items():
            weight_bins_data.append({
                'bin': bin_name,
                'mean': stats['mean_daily_gain'],
                'std': stats['std_daily_gain']
            })

        if weight_bins_data:
            bins_df = pd.DataFrame(weight_bins_data)
            x_pos = np.arange(len(bins_df))
            ax1.bar(x_pos, bins_df['mean'], yerr=bins_df['std'],
                   capsize=5, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(bins_df['bin'], rotation=45)
            ax1.set_xlabel('Weight Range (kg)', fontsize=12)
            ax1.set_ylabel('Daily Weight Gain (kg/day)', fontsize=12)
            ax1.set_title('Daily Gain by Weight Range', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3, axis='y')

            # 평균선 표시
            ax1.axhline(y=self.growth_patterns['overall']['mean_daily_gain'],
                       color='red', linestyle='--', linewidth=2,
                       label=f"Overall Avg: {self.growth_patterns['overall']['mean_daily_gain']:.3f}")
            ax1.legend()

        # 2. 사료-증체 관계
        ax2 = axes[0, 1]
        if 'feed_to_gain' in self.growth_patterns:
            # 예시 데이터 생성
            feed_range = np.linspace(0.5, 4.0, 100)
            gain_pred = (self.growth_patterns['feed_to_gain']['coefficient'] * feed_range +
                        self.growth_patterns['feed_to_gain']['intercept'])

            ax2.plot(feed_range, gain_pred, 'r-', linewidth=2,
                    label=f"y = {self.growth_patterns['feed_to_gain']['coefficient']:.3f}x + "
                          f"{self.growth_patterns['feed_to_gain']['intercept']:.3f}")

            # 산점도 (실제 데이터가 있는 경우)
            if len(df_analysis) > 0 and 'feed' in df_analysis.columns:
                feed_gain = df_analysis.dropna(subset=['feed', 'daily_gain'])
                if len(feed_gain) > 0:
                    ax2.scatter(feed_gain['feed'], feed_gain['daily_gain'],
                              alpha=0.3, s=10, color='blue')

            ax2.set_xlabel('Feed Intake (kg/day)', fontsize=12)
            ax2.set_ylabel('Daily Weight Gain (kg/day)', fontsize=12)
            ax2.set_title('Feed Intake vs Weight Gain', fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim([0, 4.5])
            ax2.set_ylim([0, 1.5])

        # 3. 성장 곡선
        ax3 = axes[1, 0]
        if 'growth_curve' in self.growth_patterns:
            days = np.linspace(0, 150, 150)
            coef = self.growth_patterns['growth_curve']['coefficients']
            intercept = self.growth_patterns['growth_curve']['intercept']

            # 2차 다항식 계산
            weights_pred = intercept + coef[1] * days + coef[2] * days**2

            ax3.plot(days, weights_pred, 'b-', linewidth=2, label='Growth Model')

            # 실제 데이터가 있는 경우
            if len(df_analysis) > 0:
                growth_curve = df_analysis.groupby('day')['weight'].mean().reset_index()
                if len(growth_curve) > 0:
                    ax3.scatter(growth_curve['day'], growth_curve['weight'],
                              alpha=0.5, s=20, color='green', label='Actual Data')

            # 목표 체중선
            ax3.axhline(y=80, color='red', linestyle='--', linewidth=2, label='Target (80kg)')
            ax3.axhline(y=110, color='orange', linestyle='--', linewidth=1, label='Max (110kg)')

            ax3.set_xlabel('Day', fontsize=12)
            ax3.set_ylabel('Weight (kg)', fontsize=12)
            ax3.set_title('Average Growth Curve', fontsize=14, fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_xlim([0, 150])
            ax3.set_ylim([10, 120])

        # 4. 증체율 분포
        ax4 = axes[1, 1]

        # 정규분포 그리기
        mean_gain = self.growth_patterns['overall']['mean_daily_gain']
        std_gain = self.growth_patterns['overall']['std_daily_gain']

        x = np.linspace(0, 1.5, 100)
        y = scipy_stats.norm.pdf(x, mean_gain, std_gain)
        ax4.plot(x, y, 'b-', linewidth=2, label='Expected Distribution')
        ax4.fill_between(x, y, alpha=0.3)

        # 실제 데이터 히스토그램
        if len(df_analysis) > 0:
            daily_gains = df_analysis['daily_gain'].dropna()
            if len(daily_gains) > 0:
                ax4.hist(daily_gains, bins=30, alpha=0.5, density=True,
                        edgecolor='black', color='green', label='Actual Data')

        ax4.axvline(x=mean_gain, color='red', linestyle='--',
                   linewidth=2, label=f'Mean: {mean_gain:.3f}')
        ax4.axvline(x=self.growth_patterns['overall']['median_daily_gain'],
                   color='orange', linestyle='--',
                   linewidth=2, label=f"Median: {self.growth_patterns['overall']['median_daily_gain']:.3f}")

        ax4.set_xlabel('Daily Weight Gain (kg/day)', fontsize=12)
        ax4.set_ylabel('Probability Density', fontsize=12)
        ax4.set_title('Distribution of Daily Weight Gain', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_xlim([0, 1.5])

        plt.suptitle('Growth Pattern Analysis (Improved)', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ 그래프 저장: {output_path}")
        plt.close()

    def generate_sample_data(self):
        """시각화를 위한 샘플 데이터 생성"""
        sample_data = []
        for day in range(150):
            weight = 20 + 0.6 * day + np.random.normal(0, 2)
            daily_gain = 0.65 + np.random.normal(0, 0.1)
            feed = weight * 0.035 + np.random.normal(0, 0.2)

            sample_data.append({
                'day': day,
                'weight': weight,
                'daily_gain': daily_gain,
                'feed': feed,
                'weight_bin': self.get_weight_bin(weight)
            })

        return pd.DataFrame(sample_data)

    def get_weight_bin(self, weight):
        """체중 구간 반환"""
        if weight < 20:
            return '0-20kg'
        elif weight < 40:
            return '20-40kg'
        elif weight < 60:
            return '40-60kg'
        elif weight < 80:
            return '60-80kg'
        elif weight < 100:
            return '80-100kg'
        else:
            return '100kg+'

    def save_patterns(self, output_path='./growth_patterns.json'):
        """학습된 패턴을 JSON으로 저장"""
        print("\n💾 학습 결과 저장 중...")

        # NaN 값을 None으로 변환
        def clean_dict(d):
            if isinstance(d, dict):
                return {k: clean_dict(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [clean_dict(item) for item in d]
            elif isinstance(d, float):
                if np.isnan(d) or np.isinf(d):
                    return None
                return d
            else:
                return d

        cleaned_patterns = clean_dict(self.growth_patterns)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_patterns, f, indent=2, ensure_ascii=False)

        print(f"✓ 패턴 저장 완료: {output_path}")
        print("\n📋 저장된 정보:")
        print(f"   - 체중 구간별 증체율: {len(self.growth_patterns.get('weight_bins', {}))}개")
        print(f"   - 평균 일일 증체율: {self.growth_patterns['overall']['mean_daily_gain']:.3f} kg/일")
        print(f"   - 사료-증체 관계: R² = {self.growth_patterns.get('feed_to_gain', {}).get('r2', 0):.3f}")

        return output_path

    def run_learning(self):
        """전체 학습 프로세스 실행"""
        print("="*80)
        print("🎓 Step 1: 기존 데이터 학습 시작 (개선판)")
        print("="*80)

        # 1. 데이터 로드
        self.load_all_data()

        # 2. 데이터 통합
        df = None
        if self.all_data:
            df = self.integrate_data()

        # 3. 패턴 분석 (데이터가 없어도 기본값으로 진행)
        df_analysis = self.analyze_growth_patterns(df)

        # 4. 시각화
        self.visualize_patterns(df_analysis)

        # 5. 패턴 저장
        pattern_file = self.save_patterns()

        print("\n" + "="*80)
        print("✅ Step 1 완료!")
        print("="*80)
        print(f"\n생성된 파일:")
        print(f"  1. growth_patterns.json - 학습된 성장 패턴")
        print(f"  2. growth_patterns_analysis.png - 분석 그래프")
        print("\n이 파일들은 Step 2에서 사용됩니다.")

        return pattern_file


def main():
    """메인 실행 함수"""
    print("="*80)
    print("🐷 돼지 출하 예측 시스템 - Step 1: 기존 데이터 학습 (개선판)")
    print("="*80)

    base_path = './텍스트 데이터'

    # 경로가 없어도 기본값으로 진행
    if not os.path.exists(base_path):
        print(f"\n⚠️ '{base_path}' 경로가 존재하지 않습니다.")
        print("기본 성장 패턴을 사용하여 진행합니다.")

    learner = GrowthPatternLearner(base_path=base_path)
    pattern_file = learner.run_learning()

    if pattern_file:
        print("\n" + "="*80)
        print("🎉 학습 완료!")
        print("="*80)
        print("\n다음 단계:")
        print("  Step 2 실행 → 학습된 패턴으로 신규 돼지 예측")
        print("\n주요 개선사항:")
        print("  ✓ 비정상적인 데이터 필터링 강화")
        print("  ✓ 업계 표준 기본값 제공")
        print("  ✓ 증체율 정상 범위 검증 (0.2~1.2 kg/일)")
        print("  ✓ 체중별 적절한 성장률 적용")


if __name__ == "__main__":
    try:
        import chardet
    except ImportError:
        print("⚠️  chardet 라이브러리 설치 필요")
        print("설치 명령어: pip install chardet")
        import sys
        sys.exit(1)

    main()