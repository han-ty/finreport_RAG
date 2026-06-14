"""
PDF财报解析模块
从财务报告PDF中提取结构化数据
支持上交所和深交所两种命名格式
"""
import os
import re
import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ReportMeta:
    """财报元数据"""
    file_path: str = ""
    file_name: str = ""
    stock_code: str = ""
    stock_abbr: str = ""
    report_year: int = 0
    report_period: str = ""  # FY, Q1, HY, Q3
    report_type: str = ""  # 年度报告, 半年度报告, 一季度报告, 三季度报告, 报告摘要
    exchange: str = ""  # 上交所, 深交所
    publish_date: str = ""


@dataclass
class FinancialData:
    """提取的财务数据"""
    meta: ReportMeta = field(default_factory=ReportMeta)
    core_performance: Dict[str, Any] = field(default_factory=dict)
    balance_sheet: Dict[str, Any] = field(default_factory=dict)
    income_sheet: Dict[str, Any] = field(default_factory=dict)
    cash_flow_sheet: Dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""
    tables: List[Dict] = field(default_factory=list)


def parse_report_meta_shanghai(filename: str) -> ReportMeta:
    """
    解析上交所报告文件名
    格式：股票代码_报告日期_随机标识.pdf
    例如：600080_20230428_FQ2V.pdf
    """
    meta = ReportMeta(exchange="上交所", file_name=filename)
    parts = filename.replace(".pdf", "").split("_")
    if len(parts) >= 3:
        meta.stock_code = parts[0]
        date_str = parts[1]
        if len(date_str) == 8:
            meta.publish_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
            month = int(date_str[4:6])
            # 根据发布月份推断报告类型
            # 4月发布的通常是年报或一季报
            # 8月发布的通常是半年报
            # 10月发布的通常是三季报
            if month in [3, 4, 5]:
                # 可能是年报或一季报，需要根据内容判断
                meta.report_type = "待确定"
            elif month in [7, 8, 9]:
                meta.report_type = "半年度报告"
            elif month in [10, 11]:
                meta.report_type = "三季度报告"
    return meta


def parse_report_meta_shenzhen(filename: str) -> ReportMeta:
    """
    解析深交所报告文件名
    格式：A股简称：年份+报告周期+报告类型.pdf
    例如：华润三九：2023年年度报告.pdf
    """
    meta = ReportMeta(exchange="深交所", file_name=filename)
    name_part = filename.replace(".pdf", "")

    # 提取公司简称（冒号前的部分）
    if "：" in name_part:
        meta.stock_abbr = name_part.split("：")[0]
        report_part = name_part.split("：")[1]
    elif ":" in name_part:
        meta.stock_abbr = name_part.split(":")[0]
        report_part = name_part.split(":")[1]
    else:
        return meta

    # 提取年份
    year_match = re.search(r'(\d{4})年', report_part)
    if year_match:
        meta.report_year = int(year_match.group(1))

    # 确定报告类型和期间
    if "年度报告摘要" in report_part:
        meta.report_type = "年度报告摘要"
        meta.report_period = f"{meta.report_year}FY"
    elif "年度报告" in report_part:
        meta.report_type = "年度报告"
        meta.report_period = f"{meta.report_year}FY"
    elif "半年度报告摘要" in report_part:
        meta.report_type = "半年度报告摘要"
        meta.report_period = f"{meta.report_year}HY"
    elif "半年度报告" in report_part:
        meta.report_type = "半年度报告"
        meta.report_period = f"{meta.report_year}HY"
    elif "一季度报告" in report_part:
        meta.report_type = "一季度报告"
        meta.report_period = f"{meta.report_year}Q1"
    elif "三季度报告" in report_part:
        meta.report_type = "三季度报告"
        meta.report_period = f"{meta.report_year}Q3"

    return meta


def extract_text_from_pdf(pdf_path: str) -> str:
    """从PDF提取文本"""
    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
        return "\n".join(text_parts)
    except ImportError:
        logger.error("请安装 pdfplumber: pip install pdfplumber")
        raise


def extract_tables_from_pdf(pdf_path: str) -> List[List[List[str]]]:
    """从PDF提取表格"""
    try:
        import pdfplumber
        all_tables = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                if tables:
                    all_tables.extend(tables)
        return all_tables
    except ImportError:
        logger.error("请安装 pdfplumber: pip install pdfplumber")
        raise


_FINANCIAL_TABLE_KEYWORDS = frozenset([
    "营业收入", "净利润", "营业利润", "利润总额",
    "总资产", "总负债", "股东权益", "资产合计", "负债合计",
    "经营活动", "投资活动", "筹资活动", "现金流量",
    "每股收益", "每股净资产", "净资产收益率", "毛利率",
    "货币资金", "应收账款", "存货", "应付账款",
    "营业成本", "销售费用", "管理费用", "财务费用", "研发费用",
    "合并资产负债表", "合并利润表", "合并现金流量表",
    "主要会计数据", "主要财务指标", "财务报表",
    "扣除非经常性损益", "非经常性损益", "加权平均",
    "所有者权益", "利润表", "资产负债",
])


def extract_text_and_tables_from_pdf(pdf_path: str, max_pages: int = 60) -> tuple:
    """从PDF同时提取文本和表格（优化版）
    
    优化策略：
    1. 限制前max_pages页（财务数据通常在前30页）
    2. 仅对含财务关键词的页面提取表格（节省~70%时间）
    """
    try:
        import pdfplumber
        text_parts = []
        all_tables = []
        with pdfplumber.open(pdf_path) as pdf:
            pages_to_process = pdf.pages[:max_pages] if len(pdf.pages) > max_pages else pdf.pages
            for page_idx, page in enumerate(pages_to_process):
                text = page.extract_text()
                if text:
                    text_parts.append(text)
                    # 去空格后匹配关键词（修复PDF中字符间距导致的匹配失败）
                    text_nospace = text.replace(" ", "").replace("\u3000", "")
                    # 前15页无条件提取表格（主要会计数据/财务指标通常在此范围）
                    # 之后的页面仅对含关键词的页面提取表格
                    should_extract = (
                        page_idx < 15
                        or any(kw in text or kw in text_nospace for kw in _FINANCIAL_TABLE_KEYWORDS)
                    )
                    if should_extract:
                        tables = page.extract_tables()
                        if tables:
                            all_tables.extend(tables)
        return "\n".join(text_parts), all_tables
    except ImportError:
        logger.error("请安装 pdfplumber: pip install pdfplumber")
        raise


def classify_report_by_content(text: str, meta: ReportMeta) -> ReportMeta:
    """
    根据PDF内容进一步确定报告类型
    主要用于上交所格式（文件名不含报告类型信息）
    """
    text_lower = text[:5000]  # 只看前面部分

    if "年度报告摘要" in text_lower:
        meta.report_type = "年度报告摘要"
    elif "年度报告" in text_lower or "年 度 报 告" in text_lower:
        meta.report_type = "年度报告"
    elif "半年度报告摘要" in text_lower:
        meta.report_type = "半年度报告摘要"
    elif "半年度报告" in text_lower or "半 年 度 报 告" in text_lower:
        meta.report_type = "半年度报告"
    elif "第一季度报告" in text_lower or "一季度报告" in text_lower:
        meta.report_type = "一季度报告"
    elif "第三季度报告" in text_lower or "三季度报告" in text_lower:
        meta.report_type = "三季度报告"

    # 提取年份
    year_match = re.search(r'(20\d{2})\s*年', text_lower)
    if year_match and meta.report_year == 0:
        meta.report_year = int(year_match.group(1))

    # 确定report_period
    if meta.report_year > 0 and not meta.report_period:
        period_map = {
            "年度报告": "FY", "年度报告摘要": "FY",
            "半年度报告": "HY", "半年度报告摘要": "HY",
            "一季度报告": "Q1", "三季度报告": "Q3",
        }
        suffix = period_map.get(meta.report_type, "")
        if suffix:
            meta.report_period = f"{meta.report_year}{suffix}"

    # 提取股票代码和简称
    if not meta.stock_abbr:
        abbr_match = re.search(r'(?:股票简称|A股简称)[：:]\s*(\S+)', text_lower)
        if abbr_match:
            meta.stock_abbr = abbr_match.group(1).strip()

    code_match = re.search(r'(?:股票代码|证券代码)[：:]\s*(\d{6})', text_lower)
    if code_match and not meta.stock_code:
        meta.stock_code = code_match.group(1)

    return meta


# =============================================================================
# 财务数据规则提取器
# =============================================================================

def _parse_number(text: str) -> Optional[float]:
    """解析数字字符串为浮点数"""
    if not text or text.strip() in ["-", "—", "–", "N/A", "不适用", ""]:
        return None
    text = text.strip().replace(",", "").replace("，", "").replace(" ", "")
    # 处理百分号
    is_percent = "%" in text
    text = text.replace("%", "")
    # 处理括号表示负数
    if text.startswith("(") and text.endswith(")"):
        text = "-" + text[1:-1]
    if text.startswith("（") and text.endswith("）"):
        text = "-" + text[1:-1]
    try:
        val = float(text)
        return val
    except ValueError:
        # 尝试从文本中提取数字（如 "1234.56元" 或 "12.5%"）
        match = re.search(r'[-]?[\d]+\.?\d*', text)
        if match:
            try:
                return float(match.group())
            except ValueError:
                pass
        return None


def _extract_value_from_text(text: str, patterns: List[str]) -> Optional[float]:
    """使用正则模式从文本中提取数值"""
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            val = _parse_number(match.group(1))
            if val is not None:
                return val
    return None


def _extract_from_text(text: str, keywords: List[str]) -> Optional[float]:
    """从文本中用关键字+正则提取数值（表格提取的补充）
    
    处理常见的文本布局：
    - 营业收入 1,234,567,890.00
    - 营业收入（元） 1,234,567,890.00
    - 营业收入    123,456.78   110,000.00
    """
    text_nospace = text.replace(" ", "").replace("\u3000", "")
    for kw in keywords:
        # Pattern 1: 关键字后面紧跟数字（去空格后）
        p1 = re.escape(kw) + r'[\(（][^)）]*[\)）]\s*([-\(（]?[\d,，]+\.?\d*[\)）]?)'
        m = re.search(p1, text_nospace)
        if m:
            val = _parse_number(m.group(1))
            if val is not None:
                return val
        # Pattern 2: 关键字后面有空格然后数字（原始文本中）
        for kw2 in [kw, kw.replace(" ", "")]:
            p2 = re.escape(kw2) + r'[\s\(（）\)元万]*?\s{2,}([-\(（]?[\d,，]+\.?\d*[\)）]?)'
            m = re.search(p2, text)
            if m:
                val = _parse_number(m.group(1))
                if val is not None:
                    return val
        # Pattern 3: 去空格文本中的匹配
        p3 = re.escape(kw) + r'[^\d\n]{0,20}?([-\(（]?[\d,，]+\.?\d*[\)）]?)'
        m = re.search(p3, text_nospace)
        if m:
            val = _parse_number(m.group(1))
            if val is not None:
                return val
    return None


def extract_financial_data_by_rules(
    text: str,
    tables: List[List[List[str]]],
    meta: ReportMeta,
) -> FinancialData:
    """
    通过规则从文本和表格中提取财务数据
    这是第一步，后续可以用LLM增强
    """
    data = FinancialData(meta=meta, raw_text=text, tables=tables)

    # 提取利润表相关数据
    data.income_sheet = _extract_income_data(text, tables)
    # 提取资产负债表相关数据
    data.balance_sheet = _extract_balance_data(text, tables)
    # 提取现金流量表相关数据
    data.cash_flow_sheet = _extract_cash_flow_data(text, tables)
    # 提取核心业绩指标
    data.core_performance = _extract_core_performance(text, tables)

    return data


def _find_table_value(tables: List, row_keywords: List[str], col_index: int = None) -> Optional[float]:
    """在表格中查找特定行的值
    
    改进：
    1. 尝试多列（1~4），找到第一个有效数值
    2. 去空格匹配（处理"营 业 收 入"等情况）
    3. 检查前2列作为关键字列
    4. 优先返回本期金额（第一个有效数字列）
    """
    for table in tables:
        for row in table:
            if not row:
                continue
            # 检查前2列是否包含关键字
            for kw_col in range(min(2, len(row))):
                cell = str(row[kw_col] or "").strip()
                cell_nospace = cell.replace(" ", "").replace("\u3000", "")
                matched = False
                for kw in row_keywords:
                    if kw in cell or kw in cell_nospace:
                        matched = True
                        break
                if not matched:
                    continue
                # 指定列优先
                if col_index is not None and len(row) > col_index:
                    val = _parse_number(str(row[col_index] or ""))
                    if val is not None:
                        return val
                # 依次尝试关键字列之后的列
                for ci in range(kw_col + 1, min(len(row), kw_col + 5)):
                    val = _parse_number(str(row[ci] or ""))
                    if val is not None:
                        return val
    return None


def _extract_with_fallback(tables: List, text: str, field_mappings: Dict[str, List[str]]) -> Dict[str, Any]:
    """通用提取函数：先表格，再文本，确保最大覆盖"""
    data = {}
    for field_name, keywords in field_mappings.items():
        # 先从表格提取
        val = _find_table_value(tables, keywords)
        if val is None:
            # 表格没找到，从文本提取
            val = _extract_from_text(text, keywords)
        if val is not None:
            data[field_name] = val
    return data


def _extract_income_data(text: str, tables: List) -> Dict[str, Any]:
    """提取利润表数据"""
    field_mappings = {
        "total_operating_revenue": ["营业总收入", "营业收入", "一、营业总收入", "一、营业收入"],
        "operating_expense_cost_of_sales": ["营业成本", "营业支出", "减：营业成本"],
        "operating_expense_selling_expenses": ["销售费用"],
        "operating_expense_administrative_expenses": ["管理费用"],
        "operating_expense_financial_expenses": ["财务费用"],
        "operating_expense_rnd_expenses": ["研发费用"],
        "operating_expense_taxes_and_surcharges": ["税金及附加"],
        "total_operating_expenses": ["营业总支出", "营业总成本", "二、营业总成本"],
        "operating_profit": ["营业利润", "三、营业利润"],
        "total_profit": ["利润总额", "四、利润总额"],
        "net_profit": ["净利润", "五、净利润", "归属于母公司所有者的净利润", "归属于上市公司股东的净利润"],
        "other_income": ["其他收益"],
        "investment_income": ["投资收益", "对联营企业和合营企业的投资收益"],
        "fair_value_change_income": ["公允价值变动收益", "公允价值变动净收益"],
        "asset_disposal_income": ["资产处置收益"],
        "non_operating_income": ["营业外收入"],
        "non_operating_expenses": ["营业外支出"],
        "asset_impairment_loss": ["资产减值损失"],
        "credit_impairment_loss": ["信用减值损失"],
    }
    return _extract_with_fallback(tables, text, field_mappings)


def _extract_balance_data(text: str, tables: List) -> Dict[str, Any]:
    """提取资产负债表数据"""
    field_mappings = {
        "asset_cash_and_cash_equivalents": ["货币资金"],
        "asset_accounts_receivable": ["应收账款"],
        "asset_inventory": ["存货"],
        "asset_trading_financial_assets": ["交易性金融资产"],
        "asset_construction_in_progress": ["在建工程"],
        "asset_total_assets": ["资产总计", "资产合计", "总资产"],
        "liability_accounts_payable": ["应付账款"],
        "liability_advance_from_customers": ["预收账款", "预收款项"],
        "liability_total_liabilities": ["负债合计", "负债总计", "总负债"],
        "liability_contract_liabilities": ["合同负债"],
        "liability_short_term_loans": ["短期借款"],
        "equity_unappropriated_profit": ["未分配利润"],
        "equity_total_equity": [
            "所有者权益合计", "股东权益合计", "所有者权益（或股东权益）合计",
            "归属于母公司所有者权益合计", "归属于母公司股东权益合计",
            "负债和所有者权益总计", "负债和股东权益总计",
        ],
    }
    data = _extract_with_fallback(tables, text, field_mappings)

    # equity_total_equity 特殊处理：如果"负债和所有者权益总计"等于总资产，则用 总资产-总负债
    if "equity_total_equity" not in data:
        total_a = data.get("asset_total_assets")
        total_l = data.get("liability_total_liabilities")
        if total_a is not None and total_l is not None:
            data["equity_total_equity"] = round(total_a - total_l, 4)

    # 计算资产负债率
    if "asset_total_assets" in data and "liability_total_liabilities" in data:
        if data["asset_total_assets"] != 0:
            data["asset_liability_ratio"] = round(
                data["liability_total_liabilities"] / data["asset_total_assets"] * 100, 4
            )

    return data


def _extract_cash_flow_data(text: str, tables: List) -> Dict[str, Any]:
    """提取现金流量表数据"""
    field_mappings = {
        "operating_cf_cash_from_sales": [
            "销售商品、提供劳务收到的现金", "销售商品收到的现金",
            "销售商品、提供劳务收到", "销售商品提供劳务收到的现金",
        ],
        "operating_cf_net_amount": [
            "经营活动产生的现金流量净额", "经营活动现金流量净额",
        ],
        "investing_cf_net_amount": [
            "投资活动产生的现金流量净额", "投资活动现金流量净额",
        ],
        "investing_cf_cash_for_investments": ["投资支付的现金", "购建固定资产"],
        "investing_cf_cash_from_investment_recovery": ["收回投资收到的现金"],
        "financing_cf_cash_from_borrowing": ["取得借款收到的现金"],
        "financing_cf_cash_for_debt_repayment": ["偿还债务支付的现金"],
        "financing_cf_net_amount": [
            "筹资活动产生的现金流量净额", "融资活动产生的现金流量净额",
            "筹资活动现金流量净额",
        ],
        "net_cash_flow": [
            "现金及现金等价物净增加额", "五、现金及现金等价物净增加额",
            "现金及现金等价物的净增加额",
        ],
    }
    data = _extract_with_fallback(tables, text, field_mappings)

    # 计算占比
    net_cf = data.get("net_cash_flow")
    if net_cf and net_cf != 0:
        for key, ratio_key in [
            ("operating_cf_net_amount", "operating_cf_ratio_of_net_cf"),
            ("investing_cf_net_amount", "investing_cf_ratio_of_net_cf"),
            ("financing_cf_net_amount", "financing_cf_ratio_of_net_cf"),
        ]:
            if key in data:
                data[ratio_key] = round(data[key] / net_cf * 100, 4)

    return data


def _extract_core_performance(text: str, tables: List) -> Dict[str, Any]:
    """提取核心业绩指标"""
    field_mappings = {
        "eps": [
            "基本每股收益", "每股收益",
            "基本每股收益(元/股)", "基本每股收益（元/股）",
        ],
        "net_asset_per_share": [
            "每股净资产", "归属于上市公司股东的每股净资产",
            "每股净资产(元)", "每股净资产（元）",
        ],
        "roe": [
            "加权平均净资产收益率", "净资产收益率", "全面摊薄净资产收益率",
        ],
        "operating_cf_per_share": [
            "每股经营活动产生的现金流量净额", "每股经营现金流量净额",
            "每股经营现金流量", "每股经营活动现金流量净额",
            "每股经营活动产生的现金流量",
        ],
        "gross_profit_margin": ["销售毛利率", "毛利率"],
        "net_profit_excl_non_recurring": [
            "扣除非经常性损益后的净利润", "扣非净利润",
            "扣除非经常性损益", "归属于上市公司股东的扣除非经常性损益的净利润",
        ],
        "roe_weighted_excl_non_recurring": [
            "扣除非经常性损益后的加权平均净资产收益率",
            "扣非后加权平均净资产收益率",
        ],
    }
    return _extract_with_fallback(tables, text, field_mappings)


def scan_report_files(data_dir: str) -> List[ReportMeta]:
    """
    扫描数据目录，返回所有财报文件的元数据列表
    """
    reports = []
    data_path = Path(data_dir)

    # 扫描附件2
    att2_patterns = ["附件2", "财务报告"]
    for d in data_path.iterdir():
        if d.is_dir() and any(p in d.name for p in att2_patterns):
            # 扫描上交所
            sh_dir = d / "reports-上交所"
            if sh_dir.exists():
                for f in sh_dir.glob("*.pdf"):
                    meta = parse_report_meta_shanghai(f.name)
                    meta.file_path = str(f)
                    reports.append(meta)

            # 扫描深交所
            sz_dir = d / "reports-深交所"
            if sz_dir.exists():
                for f in sz_dir.glob("*.pdf"):
                    meta = parse_report_meta_shenzhen(f.name)
                    meta.file_path = str(f)
                    reports.append(meta)

    logger.info(f"扫描到 {len(reports)} 个财报文件")
    return reports
