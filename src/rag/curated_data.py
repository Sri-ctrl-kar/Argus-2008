"""
Verified, High-Fidelity 10-K Disclosures for Target Benchmark Companies.
Contains factual financial numbers, risk factors, and MD&A details for AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, AMD, INTC, NFLX.
"""

from typing import List, Dict, Any


def get_curated_filings_data() -> List[Dict[str, Any]]:
    return [
        {
            "ticker": "AAPL",
            "company_name": "Apple Inc.",
            "cik": "0000320193",
            "fiscal_year": 2023,
            "filing_date": "2023-11-03",
            "sections": {
                "ITEM_1": """Item 1. Business
Apple Inc. designs, manufactures and markets smartphones, personal computers, tablets, wearables and accessories, and sells a variety of related services. 
The Company's product lines include iPhone, Mac, iPad, and Wearables, Home and Accessories (comprising AirPods, Apple TV, Apple Watch, Beats products, and HomePod). 
Apple's Services segment includes the App Store, Apple Music, Apple Pay, AppleCare, Apple TV+, Apple Arcade, Apple Card, and iCloud. 
The Company operates retail stores, online stores, and direct sales forces, alongside third-party cellular network carriers, wholesalers, and retailers. 
The Company's fiscal year is the 52- or 53-week period that ends on the last Saturday of September.""",
                "ITEM_1A": """Item 1A. Risk Factors
The Company's business, results of operations and financial condition can be adversely affected by several factors:
1. Global and regional economic conditions, including inflation, high interest rates, currency fluctuations, and geopolitical tensions.
2. Global supply chain constraints, component shortages, and single-source dependencies. Substantially all of the Company's manufacturing is performed by outsourced partners located primarily in Asia, including mainland China, India, Vietnam, and Taiwan.
3. Rapid technological change and intense competition in highly competitive markets characterized by short product life cycles.
4. Regulatory scrutiny and legal proceedings concerning digital marketplaces, the App Store commission structure, and antitrust investigations in the US and European Union (such as the Digital Markets Act).
5. Cybersecurity vulnerabilities, data privacy regulations, and potential intellectual property infringement claims.""",
                "ITEM_7": """Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations
Fiscal Year 2023 Highlights:
Total net sales decreased 2.8% to $383,285 million in FY2023 compared to $394,328 million in FY2022.
- iPhone net sales were $200,583 million compared to $205,489 million in FY2022.
- Mac net sales were $29,357 million, down 27% from $40,177 million in FY2022, driven by challenging market dynamics in personal computers.
- iPad net sales were $28,300 million compared to $29,292 million in FY2022.
- Wearables, Home and Accessories net sales were $39,845 million compared to $41,241 million in FY2022.
- Services net sales reached an all-time record of $85,200 million, up 9% from $78,129 million in FY2022, driven by growth in advertising, cloud services, and payment services.
Research and Development (R&D) expense was $29,915 million in FY2023, representing 7.8% of total net sales, compared to $26,251 million in FY2022. The increase was driven primarily by headcount-related expenses and engineering development costs.""",
                "ITEM_8": """Item 8. Financial Statements and Supplementary Data
Consolidated Statements of Operations (in millions, except per share amounts):
- Total net sales: $383,285
- Total cost of sales: $214,137
- Gross margin: $169,148 (Gross margin percentage: 44.1%)
- Operating expenses: R&D $29,915; SG&A $24,932; Total operating expenses: $54,847
- Operating income: $114,301
- Other income/(expense), net: $(382)
- Income before provision for income taxes: $113,743
- Provision for income taxes: $16,741
- Net income: $96,995
- Earnings per share (diluted): $6.13
- Cash and cash equivalents: $29,965 million; Marketable securities: $132,143 million; Total debt: $111,088 million."""
            }
        },
        {
            "ticker": "MSFT",
            "company_name": "Microsoft Corp",
            "cik": "0000789019",
            "fiscal_year": 2023,
            "filing_date": "2023-08-01",
            "sections": {
                "ITEM_1": """Item 1. Business
Microsoft Corporation is a technology company that develops and supports software, services, devices, and solutions. 
The Company operates three segments:
1. Productivity and Business Processes (Office Commercial, Office Consumer, LinkedIn, and Dynamics business solutions).
2. Intelligent Cloud (Server products and cloud services including Microsoft Azure, Windows Server, SQL Server, Visual Studio, and Enterprise Services).
3. More Personal Computing (Windows OEM and commercial licensing, Devices including Surface, Gaming including Xbox hardware and Xbox content, and Search and news advertising).""",
                "ITEM_1A": """Item 1A. Risk Factors
Significant risks include:
1. Intense competition across all cloud, software, productivity, and gaming markets against competitors like Amazon Web Services (AWS) and Google Cloud.
2. Execution risks relating to massive capital investments in Artificial Intelligence (AI) infrastructure, high-performance GPUs, and datacenter buildouts.
3. Complex cybersecurity threats and supply chain dependencies for datacenter hardware.
4. Regulatory scrutiny regarding mergers and acquisitions, including the Activision Blizzard transaction and international antitrust reviews.
5. Foreign exchange volatility and global macroeconomic softening affecting commercial enterprise IT budgets.""",
                "ITEM_7": """Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations
Fiscal Year 2023 Performance:
Total revenue was $211,915 million, an increase of 7% (11% in constant currency) compared to $198,270 million in FY2022.
- Productivity and Business Processes revenue grew 8% to $69,274 million. Office 365 Commercial revenue grew 13%.
- Intelligent Cloud revenue increased 17% to $87,907 million. Azure and other cloud services revenue grew 27% (34% in constant currency).
- More Personal Computing revenue decreased 9% to $54,734 million, driven by PC market weakness impacting Windows OEM revenue.
Research and Development (R&D) expense was $27,195 million in FY2023, up 11% compared to $24,512 million in FY2022, primarily driven by investments in cloud engineering and AI technologies.""",
                "ITEM_8": """Item 8. Financial Statements and Supplementary Data
Consolidated Income Statements (in millions, except per share amounts):
- Revenue: $211,915
- Cost of revenue: $65,863
- Gross margin: $146,052 (68.9%)
- Research and development: $27,195
- Sales and marketing: $22,759
- General and administrative: $7,575
- Operating income: $88,523
- Net income: $72,361
- Diluted earnings per share: $9.68
- Total Cash, cash equivalents and short-term investments: $111,262 million."""
            }
        },
        {
            "ticker": "NVDA",
            "company_name": "NVIDIA Corp",
            "cik": "0001045810",
            "fiscal_year": 2024,
            "filing_date": "2024-02-21",
            "sections": {
                "ITEM_1": """Item 1. Business
NVIDIA Corporation pioneered GPU-accelerated computing. The Company focuses on solutions for generative AI, high-performance computing (HPC), graphics, robotics, and automotive.
The Company operates in two primary segments:
1. Compute & Networking: Includes Data Center accelerated computing platforms (Hopper architecture, H100, H200, DGX systems), Quantum InfiniBand and Spectrum Ethernet networking, and AI enterprise software (NVIDIA AI Enterprise).
2. Graphics: Includes GeForce GPUs for gaming and PCs, GeForce NOW cloud gaming, Quadro/NVIDIA RTX GPUs for enterprise workstations, and automotive cockpit computing.""",
                "ITEM_1A": """Item 1A. Risk Factors
Key risks confronting NVIDIA:
1. Severe concentration in advanced semiconductor manufacturing and packaging partners, notably Taiwan Semiconductor Manufacturing Company (TSMC) for silicon wafer fabrication and advanced CoWoS (Chip-on-Wafer-on-Substrate) packaging.
2. US government export restrictions and licensing requirements on high-performance GPUs shipped to China and other geopolitical destinations (e.g., restrictions on A100, H100, A800, H800).
3. Extreme demand fluctuations and customer concentration among hyperscale cloud service providers (Microsoft, Amazon, Google, Meta).
4. Rapid evolution of competitive architectures including custom ASICs (Google TPU, Amazon Trainium, Meta MTIA) and rival GPU vendors like AMD.""",
                "ITEM_7": """Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations
Fiscal Year 2024 Results:
Total revenue for FY2024 was $60,922 million, up 126% compared to $26,974 million in FY2023.
- Data Center revenue surged 217% to a record $47,525 million, driven by intense demand for the NVIDIA HGX platform and Hopper architecture GPUs from cloud service providers, enterprise software companies, and consumer internet companies.
- Gaming revenue was $10,447 million, up 15% from $9,067 million in FY2023.
- Professional Visualization revenue was $1,553 million, up 1%.
- Automotive revenue reached $1,091 million, up 21%.
Gross margin expanded to 72.7% compared to 56.9% in FY2023, reflecting higher Data Center sales volume and software mix.
Research and Development (R&D) expense was $8,675 million, up 18% compared to $7,339 million in FY2023, reflecting increased compensation and compute infrastructure investments.""",
                "ITEM_8": """Item 8. Financial Statements and Supplementary Data
Consolidated Statements of Income (in millions, except per share amounts):
- Revenue: $60,922
- Cost of revenue: $16,621
- Gross profit: $44,301 (72.7%)
- Operating expenses: R&D $8,675; SG&A $2,654; Total operating expenses: $11,329
- Operating income: $32,972 (up 681% from $4,224 million in FY2023)
- Net income: $29,760 (up 581% from $4,368 million in FY2023)
- Diluted earnings per share: $11.93
- Cash, cash equivalents and marketable securities: $25,984 million; Total debt: $11,056 million."""
            }
        },
        {
            "ticker": "AMZN",
            "company_name": "Amazon.com Inc",
            "cik": "0001018724",
            "fiscal_year": 2023,
            "filing_date": "2024-02-02",
            "sections": {
                "ITEM_1": """Item 1. Business
Amazon.com, Inc. serves consumers through retail websites and physical stores, with focus on selection, price, and convenience. 
The Company manufactures and sells electronic devices (Kindle, Fire tablets, Echo, Ring), produces media content, and provides Amazon Prime membership programs. 
The Company provides Amazon Web Services (AWS), offering compute, storage, database, analytics, machine learning, and AI services to developers, enterprises, and government agencies. 
Segments are: North America, International, and AWS.""",
                "ITEM_1A": """Item 1A. Risk Factors
1. Intense competition across retail, e-commerce, cloud computing, logistics, advertising, and digital streaming.
2. Substantial expansion of fulfillment network and datacenter infrastructure creating high fixed costs.
3. System interruptions, cybersecurity incidents, and reliance on third-party telecommunications networks.
4. Foreign exchange risks and international trade regulatory complexities.
5. Regulatory scrutiny regarding third-party seller marketplace practices, labor relations, and privacy legislation.""",
                "ITEM_7": """Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations
Fiscal Year 2023 Financial Summary:
Total net sales increased 12% to $574,785 million in FY2023, compared with $513,983 million in FY2022.
- North America segment sales increased 12% to $352,828 million.
- International segment sales increased 11% to $131,200 million.
- AWS segment sales increased 13% to $90,757 million, compared with $80,096 million in FY2022.
Operating income was $36,852 million in FY2023, compared with $12,248 million in FY2022. AWS operating income was $24,631 million, representing 66.8% of Amazon's total operating profit.
Technology and content expense was $85,622 million in FY2023 compared to $73,213 million in FY2022.""",
                "ITEM_8": """Item 8. Financial Statements and Supplementary Data
Consolidated Statements of Operations (in millions):
- Total net sales: $574,785
- Operating expenses: Cost of sales $304,527; Fulfillment $84,664; Technology and content $85,622; Sales and marketing $44,370; General and administrative $11,811
- Operating income: $36,852
- Total non-operating income (expense): $(377)
- Net income: $30,425 (compared to net loss of $2,722 million in FY2022)
- Diluted EPS: $2.90
- Cash, cash equivalents and restricted cash: $73,890 million."""
            }
        },
        {
            "ticker": "GOOGL",
            "company_name": "Alphabet Inc.",
            "cik": "0001652044",
            "fiscal_year": 2023,
            "filing_date": "2024-01-31",
            "sections": {
                "ITEM_1": """Item 1. Business
Alphabet Inc. is a holding company whose largest subsidiary is Google. 
Google services include Search, YouTube, Google Maps, Google Play, Android, Chrome, and Google Hardware (Pixel devices). 
Google Cloud provides enterprise-grade cloud services, including Google Cloud Platform (GCP) for infrastructure and data analytics, and Google Workspace collaboration tools. 
Other Bets segment includes early-stage healthcare and autonomous driving businesses (Waymo, Verily).""",
                "ITEM_1A": """Item 1A. Risk Factors
1. Substantial revenues generated from online advertising, which is sensitive to macroeconomic cycles, advertiser spending cuts, and changes in ad privacy technologies.
2. Competitive pressure in Generative AI and search engines from rival foundation models and search integrations.
3. Complex antitrust litigation and regulatory actions across the US DOJ and European Commission regarding search distribution and advertising technology.
4. Datacenter power, custom silicon (TPU), and technical infrastructure scaling challenges.
5. Intellectual property disputes and compliance with global content moderation and privacy laws.""",
                "ITEM_7": """Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations
Fiscal Year 2023 Overview:
Alphabet consolidated revenues were $307,394 million, up 9% compared to $282,836 million in FY2022.
- Google Search & other advertising revenue was $175,033 million, up 8%.
- YouTube ads revenue was $31,510 million, up 8%.
- Google Network revenue was $31,311 million, down 5%.
- Google Subscriptions, platforms, and devices revenue was $34,688 million, up 19%.
- Google Cloud revenue reached $33,088 million, up 26% from $26,280 million in FY2022, achieving full-year operating profitability of $1,719 million compared to an operating loss of $2,975 million in FY2022.
Research and Development (R&D) expense was $45,427 million in FY2023, representing 14.8% of revenues.""",
                "ITEM_8": """Item 8. Financial Statements and Supplementary Data
Consolidated Statements of Income (in millions, except per share amounts):
- Revenues: $307,394
- Costs and expenses: Cost of revenues $133,332; R&D $45,427; Sales and marketing $27,511; General and administrative $16,799
- Operating income: $84,293 (Operating margin: 27.4%)
- Other income (expense), net: $1,424
- Net income: $73,795
- Diluted EPS: $5.80
- Total cash, cash equivalents and marketable securities: $110,916 million."""
            }
        },
        {
            "ticker": "META",
            "company_name": "Meta Platforms, Inc.",
            "cik": "0001326801",
            "fiscal_year": 2023,
            "filing_date": "2024-02-02",
            "sections": {
                "ITEM_1": """Item 1. Business
Meta Platforms, Inc. builds technologies that help people connect, find communities, and grow businesses. 
The Company operates in two segments:
1. Family of Apps (FoA): Includes Facebook, Instagram, Messenger, WhatsApp, and Threads. Substantially all revenue is generated from digital advertising.
2. Reality Labs (RL): Includes augmented, virtual, and mixed reality hardware, software, and content (Meta Quest headsets, Ray-Ban Meta smart glasses, Horizon social platform).""",
                "ITEM_1A": """Item 1A. Risk Factors
1. Advertising revenues are vulnerable to platform privacy changes (such as Apple iOS App Tracking Transparency), macroeconomic volatility, and shifts in user engagement toward short-form video (Reels).
2. Large ongoing operating losses in Reality Labs segment resulting from multi-year investments in metaverse hardware and spatial computing.
3. Intensified antitrust, child safety, and data privacy regulatory investigations worldwide.
4. Heavy capital expenditure commitments for AI compute clusters, customized silicon (MTIA), and datacenter expansions.
5. Intense competition for user attention from platforms like TikTok, YouTube, and X.""",
                "ITEM_7": """Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations
Fiscal Year 2023 Financial Summary ("Year of Efficiency"):
Total revenue was $134,902 million, up 16% compared to $116,609 million in FY2022.
- Family of Apps revenue was $133,006 million, with advertising revenue representing $131,948 million.
- Reality Labs revenue was $1,896 million, down 12%, while incurring an operating loss of $16,120 million in FY2023 compared to an operating loss of $13,717 million in FY2022.
Total costs and expenses were $88,151 million, down 1% year-over-year, reflecting restructuring charges and head-count reductions.
Research and Development (R&D) expense was $38,482 million in FY2023, compared to $35,338 million in FY2022.""",
                "ITEM_8": """Item 8. Financial Statements and Supplementary Data
Consolidated Statements of Income (in millions, except per share amounts):
- Revenue: $134,902
- Costs and expenses: Cost of revenue $26,172; R&D $38,482; Marketing and sales $12,246; General and administrative $11,251
- Income from operations: $46,751 (Operating margin: 34.7%)
- Net income: $39,098 (up 69% from $23,200 million in FY2022)
- Diluted earnings per share: $14.87
- Cash, cash equivalents and marketable securities: $65,403 million."""
            }
        },
        {
            "ticker": "TSLA",
            "company_name": "Tesla, Inc.",
            "cik": "0001318605",
            "fiscal_year": 2023,
            "filing_date": "2024-01-29",
            "sections": {
                "ITEM_1": """Item 1. Business
Tesla, Inc. designs, develops, manufactures, sells, and leases high-performance fully electric vehicles (Model 3, Model Y, Model S, Model X, Cybertruck, Tesla Semi), solar energy generation systems, and energy storage products (Powerwall, Megapack). 
The Company also provides Full Self-Driving (FSD) capability, vehicle service centers, Supercharger fast-charging stations, and automotive insurance.""",
                "ITEM_1A": """Item 1A. Risk Factors
1. Automotive gross margin compression caused by vehicle price cuts, higher financing interest rates, and intensifying global EV competition (particularly from Chinese manufacturers like BYD).
2. Production ramp-up risks for new manufacturing architectures, Cybertruck 4680 battery cells, and new factory expansions in Texas, Berlin, and Mexico.
3. Dependence on single-source suppliers for battery cells, critical lithium/nickel minerals, and automotive semiconductors.
4. Regulatory reviews and safety investigations regarding Autopilot, FSD Beta, and vehicle recall compliance.
5. Volatility in battery raw material pricing and global supply chain logistics.""",
                "ITEM_7": """Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations
Fiscal Year 2023 Performance:
Total revenues increased 19% to $96,773 million in FY2023 compared to $81,462 million in FY2022.
- Total automotive revenues were $82,419 million, up 15%, including $1,790 million in regulatory credits.
- Energy generation and storage revenue surged 54% to $6,035 million, driven by Megapack deployment growth of 125% to 14.7 GWh.
- Services and other revenue reached $8,319 million, up 37%.
Total vehicle deliveries were 1,808,581 vehicles (1,739,707 Model 3/Y and 68,874 other models).
Automotive gross margin (excluding regulatory credits) declined to 17.1% from 26.2% in FY2022 due to vehicle price reductions.
Research and Development (R&D) expense was $3,969 million in FY2023 compared to $3,075 million in FY2022.""",
                "ITEM_8": """Item 8. Financial Statements and Supplementary Data
Consolidated Statements of Operations (in millions, except per share amounts):
- Total revenues: $96,773
- Total cost of revenues: $79,113
- Gross profit: $17,660 (Gross margin: 18.2%)
- Operating expenses: R&D $3,969; SG&A $4,800; Total operating expenses: $8,769
- Income from operations: $8,891
- Net income: $14,997 (includes $5,000 million non-cash tax valuation allowance benefit)
- Diluted EPS: $4.30
- Cash, cash equivalents and investments: $29,094 million; Total debt: $2,857 million."""
            }
        },
        {
            "ticker": "AMD",
            "company_name": "Advanced Micro Devices, Inc.",
            "cik": "0000002488",
            "fiscal_year": 2023,
            "filing_date": "2024-01-31",
            "sections": {
                "ITEM_1": """Item 1. Business
Advanced Micro Devices, Inc. is a global semiconductor company. 
The Company operates in four segments:
1. Data Center: Includes server CPUs (EPYC), data center GPUs (Instinct MI300A, MI300X), and adaptive SoC solutions (Pensando).
2. Client: Includes desktop and notebook PC microprocessors (Ryzen CPUs and APUs).
3. Gaming: Includes discrete graphics processors (Radeon GPUs) and semi-custom SoC products for gaming consoles (PlayStation 5, Xbox Series X/S).
4. Embedded: Includes FPGA, adaptive SoC, and ACAP products from the Xilinx acquisition.""",
                "ITEM_1A": """Item 1A. Risk Factors
1. Fierce competition from NVIDIA in Data Center AI accelerators and GPUs, and from Intel in server and client x86 microprocessors.
2. Reliance on third-party foundry TSMC for manufacturing wafers using advanced process nodes (5nm, 4nm, 3nm) and advanced 3D packaging.
3. Geopolitical risks relating to US export controls on advanced AI processors to China.
4. Cyclical downturns in gaming console hardware demand and enterprise embedded markets.
5. Large goodwill and intangible assets related to the Xilinx acquisition subject to potential impairment.""",
                "ITEM_7": """Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations
Fiscal Year 2023 Financial Review:
Net revenue was $22,680 million in FY2023, down 4% compared to $23,601 million in FY2022.
- Data Center segment revenue was $6,496 million, up 7% year-over-year, driven by EPYC server CPU adoption and early Instinct MI300 GPU shipments.
- Client segment revenue was $4,651 million, down 25%, due to PC supply chain inventory corrections.
- Gaming segment revenue was $6,213 million, down 9%, reflecting lower semi-custom console SoC revenue.
- Embedded segment revenue was $5,320 million, up 17%, driven by a full year of Xilinx contribution.
Research and Development (R&D) expense was $5,872 million in FY2023 compared to $5,005 million in FY2022.""",
                "ITEM_8": """Item 8. Financial Statements and Supplementary Data
Consolidated Statements of Operations (in millions, except per share amounts):
- Net revenue: $22,680
- Cost of sales: $12,242
- Gross margin: $10,438 (46.0%)
- Research and development: $5,872
- Marketing, general and administrative: $2,308
- Amortization of acquisition-related intangibles: $1,855
- Operating income: $401
- Net income: $854
- Diluted EPS: $0.53
- Cash, cash equivalents and short-term investments: $5,775 million."""
            }
        },
        {
            "ticker": "INTC",
            "company_name": "Intel Corp",
            "cik": "0000050863",
            "fiscal_year": 2023,
            "filing_date": "2024-01-26",
            "sections": {
                "ITEM_1": """Item 1. Business
Intel Corporation designs and manufactures semiconductors and computing platforms. 
Segments include:
1. Client Computing Group (CCG): PC and 2-in-1 processors (Core Ultra, Raptor Lake).
2. Data Center and AI (DCAI): Server processors (Xeon Scalable), Gaudi AI accelerators, and FPGA silicon.
3. Network and Edge (NEX): Ethernet controllers, switches, and edge computing.
4. Mobileye: Autonomous driving assistance systems.
5. Intel Foundry Services (IFS): Third-party semiconductor contract manufacturing and advanced packaging.""",
                "ITEM_1A": """Item 1A. Risk Factors
1. Execution and technological risks in the "5 Nodes in 4 Years" manufacturing roadmap (Intel 7, Intel 4, Intel 3, Intel 20A, Intel 18A).
2. Loss of market share to AMD in data center server CPUs and to ARM-based architecture in client PCs.
3. Massive capital expenditures required for foundry fab construction in Arizona, Ohio, Germany, and Ireland.
4. Subsidies and governmental grant dependencies (US CHIPS Act, European Chips Act).
5. Cyclical semiconductor demand swings and export restriction impacts in China.""",
                "ITEM_7": """Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations
Fiscal Year 2023 Overview:
Total revenue was $54,228 million in FY2023, down 14% compared to $63,054 million in FY2022.
- CCG revenue was $29,259 million, down 8%.
- DCAI revenue was $15,521 million, down 20%, due to competitive pressure in server CPUs.
- NEX revenue was $5,790 million, down 31%.
- Mobileye revenue was $2,077 million, up 11%.
- IFS revenue was $952 million, up 103%.
Research and Development (R&D) expense was $16,045 million in FY2023 compared to $17,528 million in FY2022.""",
                "ITEM_8": """Item 8. Financial Statements and Supplementary Data
Consolidated Statements of Income (in millions, except per share amounts):
- Net revenue: $54,228
- Cost of sales: $32,525
- Gross margin: $21,703 (40.0%)
- Research and development: $16,045
- Marketing, general and administrative: $5,593
- Operating income: $93 million (compared to $2,334 million in FY2022)
- Net income: $1,689 million (benefited by tax credits)
- Diluted EPS: $0.40
- Total cash, cash equivalents and short-term investments: $25,031 million; Total debt: $49,307 million."""
            }
        },
        {
            "ticker": "NFLX",
            "company_name": "Netflix, Inc.",
            "cik": "0001065280",
            "fiscal_year": 2023,
            "filing_date": "2024-01-26",
            "sections": {
                "ITEM_1": """Item 1. Business
Netflix, Inc. is one of the world's leading entertainment services, with over 260 million paid memberships in over 190 countries enjoying TV series, films, and games across a wide variety of genres and languages. 
Members can watch as much as they want, anytime, anywhere, on any internet-connected screen. 
The Company offers paid ad-supported and ad-free subscription tiers alongside paid sharing features.""",
                "ITEM_1A": """Item 1A. Risk Factors
1. Intense competition in streaming entertainment from Disney+, HBO Max/Warner Bros Discovery, Amazon Prime Video, Apple TV+, and YouTube.
2. Costs and execution risks associated with producing, licensing, and amortizing original film and television content.
3. Subscriber churn and price elasticity across international markets with varying purchasing power.
4. Foreign currency exchange rate fluctuations impacting international subscriber ARPU.
5. Reliance on third-party cloud infrastructure (Amazon Web Services) and content delivery networks (Open Connect).""",
                "ITEM_7": """Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations
Fiscal Year 2023 Review:
Revenues increased 6.7% to $33,723 million in FY2023, compared to $31,616 million in FY2022.
- Global paid memberships grew 12.8% year-over-year to 260.28 million at year-end, driven by the rollout of paid sharing and ad-tier subscriptions.
- Operating income rose 21% to $6,954 million, resulting in an operating margin of 20.6%, compared to 17.8% in FY2022.
Content amortization was $14,196 million in FY2023. Cash spent on content was $13,018 million.
Marketing expense was $2,654 million. Technology and development expense was $2,709 million.""",
                "ITEM_8": """Item 8. Financial Statements and Supplementary Data
Consolidated Statements of Operations (in millions, except per share amounts):
- Revenues: $33,723
- Cost of revenues: $19,703
- Marketing: $2,654
- Technology and development: $2,709
- General and administrative: $1,703
- Operating income: $6,954
- Net income: $5,408 (up 20% from $4,492 million in FY2022)
- Diluted EPS: $12.03
- Cash and cash equivalents: $7,117 million; Total long-term debt: $14,543 million."""
            }
        }
    ]
