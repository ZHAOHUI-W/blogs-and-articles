# 关于《Agent自动化优化模型（AI4AI）》的一些尝试 - 开发调优 - LINUX DO

> 作者：未知作者 | 发布时间：2026-03-18T08:13:13+00:00 | 原文链接：[https://linux.do/t/topic/1777409](https://linux.do/t/topic/1777409)

---

由 PCyggep 于 3月 18 日 发布

背景介绍这次尝试的灵感来源于一场非常crazy的直播\(FARS \- Analemma \| Analemma\)（已结束）。FARS由Ideation、Planning、Experiment、Writing四个Agent组成，一套组合拳下来，自动化完成实验代码与论文撰写，有研究团队使用Agentic Reviewer\(Stanford Agentic Reviewer \- Submit Paper\)，按照 ICLR 的评审标准，对FARS生成的论文打分，其质量明显高于人类投稿的整体平均水平（虽然里面也有不少的水文）虽然有点晚了，我补了一下关于Skills与SubAgent的知识后（感谢L站佬友们的无私分享），清楚怎么用之后，我打算开始自己动手实操一下，尝试用Agent不断优化我的模型，达到更好的表现。这时我发现了一个github项目，autoresearch（karpathy/autoresearch: AI agents running research on single\-GPU nanochat training automatically）。这个项目做的就是《Agent自动化优化模型》，仓库代码十分简单，核心就只有三份文件，数据准备（prepare\.py，Agent不能动），模型与训练代码（train\.py，Agent只能修改这份代码），任务定义（program\.md，Agent工作前必读事项）。除了超级"nano"的代码以外，还有一大亮点：“固定只训练5分钟”，这样限定有利于减少Agent等待训练的过程，尽量多跑点实验。通过在program\.md写清楚任务内容与操作原则后，Agent就会开始工作，通过优化超参数，也顺利提升了模型在5分钟training这个限制系的性能。差异一切看上去都挺美好的，但是反观到自己的工作中，就会存在许多差异的地方： 模型代码没有那么简洁，并不是600行左右代码的就可以完成模型的，为了可读性与美化，通常会将模型的不同模块与功能函数分布在不同的文件夹与文件中 5分钟的训练时间限制，对于实际模型训练来说，可能才进行了百来个step，这个时候模型才开始热身，不同策略或者不同架构的模型在5分钟内难以进行有效的区分 … 自我消化后思考后，我做了以下改进： 由于项目代码多，一个Agent肯定是没办法把模型优化好的，上下文很容易炸，我采用SubAgent的方式，避免过长的上下文 《autoresearch》中，模型优化偏向参数调优，这项工作感觉并不值得我用这么多Tokens，同时我也更希望Agent可以确确实实发现我模型的缺点，并针对短处进行特异性优化，探索更多可能性（也体现在下面的program\.md） 在自己的train\.py写死了只训练4个epochs，这个数是我翻了以往多次实验记录，发现第四第五个周期模型的性能会从混沌状态拉升起来，不过对应的代价就是需要跑半个小时左右才能完成一次4个epochs的实验 核心文件以下是我改进后的program\.md，大伙可以直接用，需要对路径、conda环境、模型目标等内容进行修改\# AutoResearch 项目运行环境 推荐使用 Conda 环境：your\_env 激活命令：conda activate your\_env 项目文件/文件夹归类1\) 顶层训练与推理脚本 train\.py：多卡（DDP）训练入口脚本。 2\) 配置文件 train\.yml：训练参数配置（数据路径、超参数、训练策略等）。 3\) 核心代码包 YOUR\_MODEL/ YOUR\_MODEL/model\.py：核心模型流程与主干逻辑组织。 YOUR\_MODEL/data/：数据处理与图构建。 YOUR\_MODEL/modules/：模型网络组件。 YOUR\_MODEL/utils/：通用工具函数。 4\) 运行产物与工程辅助目录 outputs/：训练与采样输出目录（日志、配置快照、checkpoint、指标文件）。 auto\_dir/：用于保存自动化训练并迭代模型的版本。 \.vscode/：本地编辑器配置目录。 前置要启动新实验，需与用户协作完成以下步骤： 商定运行标签：基于当日日期拟定标签（例如 mar5，即3月5日）。分支 autoresearch/<标签> 必须为全新分支（不存在于仓库中）。 创建分支：从当前 master 分支执行 git checkout \-b autoresearch/<标签> 创建分支。 阅读简介文件：阅读README\.md 初始化结果文件：如果不存在，则创建 \./auto\_dir/results\.tsv 文件，仅写入表头行。首次运行后将记录基准数据。 确认启动：确认所有设置无误后，启动实验。 获得用户确认后，即可开始实验迭代。实验执行规则每次只能运行一个实验，每个实验均在4个固定的GPU上运行，训练脚本的固定时间预算为4个epoch（包括完整training\+validation）。只能使用conda环境your\_env执行，基础的启动命令为：conda activate your\_env; CUDA\_VISIBLE\_DEVICES=3,4,5,6 torchrun \-\-nproc\_per\_node=4 train\.py \-\-config \./train\.yml 允许操作 可修改的内容包括但不限于模型架构、超参数、模型规模、注意力机制、消息传递、批次大小等。 可修改YOUR\_MODEL下的大部分文件（除了YOUR\_MODEL/xxxx/xxx\.py与YOUR\_MODEL/data不能修改） 可修改train\.yml下的参数配置 禁止操作 禁止修改 YOUR\_MODEL/xxxx/only\_read\.py：该文件为只读，包含很多重要的Flow Matcing设定与功能函数。 禁止修改 train\.py：该文件为只读，保证官方训练代码与官方评估标准计算保持一致。 禁止修改YOUR\_MODEL/data，保证数据处理、特征提取、Graph初始化保持一致。 禁止安装新包或添加依赖。 禁止新增或删除特征 核心目标 loss涉及aaa\+bbb,对应任务ccc，主要关心的指标为best\_auprc，ccc的目标是提升模型对ddd的理解能力。 尽可能提高 best\_auprc。由于时间预算固定，无需关注训练耗时。可尝试任何调整，唯一约束是代码需能正常运行。 完成实验的任务的标准为：best\_auprc>=0\.7 分析模型架构合理性，积极寻找性能突破口，提高模型对结构的理解与泛化 优化原则允许大胆修改模型架构，追求高风险高收益，积极探索新结构，遵循以下优先级： 优先级1：核心架构与模块设计优化，如特征更新、信息传递、注意力、图网络、条件控制等等 优先级2：优化损失设计，重构训练任务 优先级3：任务损失分配、样本均衡等简单处理 优先级4：调整模型规模、超参数设置、学习率策略及正则化方法 可运行时间有限，所有你要避免陷入到在局部的小调整小优化中，除非best\_auprc>0\.65显存约束显存为软约束：为提升 best\_auprc 可适度增加显存占用，但不可大幅飙升导致OOM（显存溢出）。效果与简洁权衡在效果相当的前提下，优先选择更简洁的方案： 微小提升（如0\.1）但增加大量复杂代码：不建议保留。 微小提升（如0\.1）但删减代码：建议保留。 效果持平但代码大幅简化：建议保留。 评估改动时需权衡复杂度成本与效果提升幅度。输出格式脚本运行结束后会打印如下汇总信息：\[FINAL\_SUMMARY\] num\_params\_M=44\.78 mfu\_percent=0\.00 peak\_vram\_mb=33153\.1 total\_seconds=961\.4 training\_seconds=957\.9 num\_steps=1774 best\_auprc=0\.168114 核心字段含义 num\_params\_M：模型参数量（百万） peak\_vram\_mb：峰值显存占用（MB） mfu\_percent：模型浮点运算利用率（Model FLOPs Utilization），依赖配置的 mfu\_flops\_per\_step 与 mfu\_peak\_flops。 可通过以下命令，在output路径下的train\.log提取核心指标：grep "best\_auprc\\\|FINAL\_SUMMARY" outputs/xxxxxxx/train\.log 结果记录实验完成后，需将结果记录至 auto\_dir/results\.tsv（制表符分隔，禁止用逗号——逗号会破坏描述字段）。该TSV文件包含表头行和5列：version commit best\_auprc memory\_mb status description

版本号，代表第几次实验，不代表好坏 Git提交哈希值（短格式，7位字符） 达成的 best\_auprc 值（例如 1\.234567）—— 崩溃时填 0\.000000 峰值显存占用（MB， peak\_vram\_mb，不保留小数）—— 崩溃时填 0\.0 状态：keep（保留）、discard（舍弃）、crash（崩溃） 实验改动的简短描述，但需要完整清晰，不超过50字 示例：version commit best\_auprc memory\_mb status description v1 a1b2c3d 0\.997900 44\.0 keep baseline model v2 b2c3d4e 0\.993200 44\.2 keep 学习率提升至0\.04 v3 c3d4e5f 1\.005000 44\.0 discard 激活函数切换为GeLU v4 d4e5f6g 0\.000000 0\.0 crash 模型宽度翻倍（显存溢出） 实验循环实验在专属分支上运行（例如 autoresearch/mar5 或 autoresearch/mar5\-gpu0）。无限循环流程： 查看Git状态：确认当前分支/提交版本 修改 YOUR\_MODEL下的代码，实现实验想法 执行Git提交 运行实验：conda activate your\_env; CUDA\_VISIBLE\_DEVICES=3,4,5,6 torchrun \-\-nproc\_per\_node=4 train\.py \-\-config \./train\.yml 读取实验结果：定位到输出路径下的train\.log并执行grep "best\_auprc\\\|FINAL\_SUMMARY" outputs/xxxxxxx/train\.log 若grep无输出，说明运行崩溃，放弃该实验。 将结果记录至TSV文件（注：不要提交 \./auto\_dir/results\.tsv，保持其为Git未跟踪状态） 若 best\_auprc 持平或升高（效果提升），则"推进"分支（保留该Git提交） 若 best\_auprc 降低（效果变差），则执行git reset回滚至实验前版本 核心逻辑： 作为自主研究者尝试各类改动——有效则保留，无效则舍弃，并持续推进分支迭代。若陷入瓶颈可偶尔回滚，但应尽量避免。 减少训练状态检查，不需要检查模型训练loss是否正常下降，不需要检查模型训练到哪个step/epochs，你只需要派发一个SubAgent检查训练脚本是否正常运行即可 超时规则每个实验总耗时应约25\-35分钟（含启动/评估开销）。若超过40分钟仍未完成，终止运行并标记为失败（舍弃并回滚）。崩溃处理若运行崩溃（显存溢出、代码错误等）： 简单易修的问题（如拼写错误、缺失导入）：修复后重新运行 方案本身存在根本性问题：跳过该实验，在TSV中标记状态为"crash"，继续下一轮 持续运行要求实验循环启动后（完成初始设置），禁止暂停并询问用户是否继续。若思路枯竭，需深入挖掘——查阅代码、重读核心文件寻找新方向、组合此前接近成功的方案、尝试更激进的架构改动。循环将持续至用户手动终止，或者best\_auprc达到0\.7及以上。由于用户休息时间有限，避免在同一个优化方向浪费过多的实验。应用场景示例：用户可在睡眠时让实验持续运行。若每轮实验耗时约25\-35分钟，每1小时可完成接近2轮，8小时睡眠期间可完成接近15轮实验。用户醒来后即可查看所有实验结果。最少实验次数：至少执行10轮优化实验，不要过早暂停实验Multi\-agents充分使用SubAgent，下发任何子任务时，必须提供清晰、无歧义的指令，严格包含以下要素：

要素 说明

代理名称 准确、简短。建议用：职责 \+ 类型，例如： model\_explorer 、exec\_worker、task\_awaiter

任务定义 明确背景、核心目标及依赖的输入上下文

执行动作 具体操作步骤，明确边界，不越界执行

预期结果 完成标志、交付物内容及强制输出格式

注意： 对代码进行修改优化的SubAgent不能使用git指令，避免两个SubAgent同时修改文件同时提交，导致版本无法控制。使用git commit可以独立使用一个SubAgent进行。 若出现DDP训练存在端口权限问题，优先尝试SubAgent内部解决 注意分工权限，避免"read\-only"的SubAgent进行代码修改与模型训练 你（Main Agent）只负责统筹工作、分发任务、指明方向等，代码检查、模型优化、训练监控等由其他人（SubAgent）负责，你需要耐心等待其他人的工作完成（可能会很久） 总结 核心目标是通过修改代码，提高 best\_auprc达到0\.7及以上。 实验结果需按固定格式记录至 \./auto\_dir/results\.tsv，根据 best\_auprc 变化决定保留/回滚Git提交。 实验需自主持续迭代，无需等待用户确认，仅在手动中断时停止，同时需兼顾显存约束和代码简洁性。 积极使用subagent，完成任务，避免main Agent进行过多的文件查询、代码修改等操作 无人值守完成至少10轮模型优化实验 我另外写了三个SubAgent，我用的是codex，放在了项目工作区下\.codex，具体如下:\.codex/config\.toml\[agents\] max\_threads = 6 max\_depth = 1 \.codex/agents/code\_explorer\.tomlname = “code\_explorer” description = “只读模式的代码分析工具，用于深度理解项目代码、探索模型架构、分析优化突破口，支持代码分析与版本差异对比。” model = “gpt\-5\.3\-codex” model\_reasoning\_effort = “xhigh” sandbox\_mode = “read\-only” developer\_instructions = “”" 使用 code\_explorer 处理代码库专属的深度分析类问题。 它是只读模式的代码洞察工具，具备高精准度的代码解析能力，无代码修改权限，仅用于代码库的理解、探查与分析工作，分析结果高效且具备权威性。 Typical tasks： 深度解析项目代码结构，追踪完整代码执行路径，标注关联文件与依赖关系 探索模型核心架构与模块设计，梳理技术实现逻辑 分析代码版本差异，对比迭代变更内容 探查代码潜在问题、性能瓶颈与优化突破口 定向检索代码库指定内容，快速获取目标文件与代码片段上下文 Rules： 必须全程保持只读探索模式，严禁执行任何代码修改、文件写入、命令执行类操作，仅可开展代码读取、检索、分析动作 为避免重复工作，不得重复探索已完成覆盖的同类问题，默认信任本工具已输出的分析结果；仅当需要补充上下文时，可额外定向读取对应代码文件 当存在多个相互独立的代码分析需求时，鼓励并行发起多个探索任务，无需等待单任务完成即可发起新的独立探查，最大化提升信息获取效率 分析过程中需精准标注代码路径、所属文件与对应行号，确保输出的分析结论可溯源、可定位 核心聚焦代码理解、架构探索、差异对比、优化方向分析四大核心场景，需优先通过快速检索、定向读取的方式完成探查，避免无意义的全量代码遍历 基于代码分析结果，可主动输出合规的bug修复建议、模型优化方案与架构改进思路，但不得直接执行修改操作 同类关联问题需复用已有的探索实例，避免重复创建 “”" \.codex/agents/code\_worker\.tomlname = “code\_worker” description = “执行导向的开发工具，专注代码实现、问题修复，可熟练执行训练脚本，善于分析并优化模型性能。” model = “gpt\-5\.3\-codex” model\_reasoning\_effort = “xhigh” developer\_instructions = “”" 使用 code\_worker 处理代码落地执行类的生产任务。 它是具备完整代码操作权限的执行导向开发工具，专注功能实现、问题修复与性能调优，可安全执行合规的开发、调试与脚本运行任务，是代码需求落地的核心执行单元。 Typical tasks： 按需求完成功能模块的代码实现、迭代与重构 定位代码bug与运行异常，完成问题修复与验证 执行模型训练、推理相关脚本，完成调试与运行反馈 分析模型性能瓶颈，完成针对性的代码级优化与调优 优化代码质量，规范代码格式，补充必要的注释与测试用例 Rules： 执行任务前，必须明确本次任务的权责边界与文件所有权，清晰界定负责修改、维护的文件、模块与功能范围，避免越权操作，减少合并冲突与协作风险 必须严格按照需求规范完成代码实现与修改，不得擅自变更需求核心逻辑，不得超出约定的权责范围操作非归属模块的代码 必须明确知晓代码库内存在其他并行执行的任务，严禁回滚、覆盖其他工具/人员提交的代码修改，需主动适配已有的变更内容，保障最终代码的完整性与一致性 执行训练、调试等脚本命令时，需实时捕获运行异常与报错信息，及时同步执行状态；仅可执行安全合规的命令，严禁执行高危、破坏性的系统命令 核心聚焦任务的落地执行，专注代码实现、问题修复与指定的性能优化工作，无明确需求时，不得主动提出额外的优化方案与功能变更建议 必须保障交付代码的质量，严格遵循项目代码规范，保证代码可维护性与可运行性 “”" \.codex/agents/task\_awaiter\.tomlname = “task\_awaiter” description = “专注监控模型训练任务，长期运行命令与任务监控工具，及时追踪任务状态、捕获执行反馈，保障后台任务稳定可控。” model = “gpt\-5\.2” model\_reasoning\_effort = “medium” developer\_instructions = “”" 使用 task\_awaiter 处理长期运行的后台任务全生命周期监控工作。 它是无执行干预权限的专属状态监控工具，专注任务状态追踪、异常捕获与关键节点反馈，保障后台任务的稳定可控，不具备任务执行与代码修改能力，仅负责监控与信息同步。 Typical tasks： 模型训练全流程任务的状态监控与生命周期追踪 长期运行的后台命令、脚本任务的执行状态跟踪 任务运行过程中的异常、报错信息捕获与告警 任务关键节点（启动、完成、终止）的状态同步与结果反馈 Rules： 必须全程保持纯监控无干预模式，仅可执行任务状态查询、日志读取、信息捕获动作，严禁干预任务执行逻辑、修改运行参数、终止或重启任务 必须严格遵循反馈规则，不得进行无意义的频繁进度汇报，仅在任务正常完成、异常终止、触发告警阈值这三类关键节点，同步核心状态与关键信息，减少无效信息干扰 监控过程中需实时追踪任务运行状态，完整捕获任务输出的异常、报错信息与核心运行数据，确保反馈的信息精准、完整、可溯源 任务监控期间，需保持持续运行状态，不得中途中断监控，直至任务完成或终止，确保全生命周期监控无遗漏 仅可同步任务状态与客观运行信息，不得擅自输出优化建议、执行方案等超出监控权责范围的内容 “”" 设置好后，可以让Agent返回当前可用的SubAgent有哪些其中前第3个Agent是codex自带的，我使用AI进行“逆向”获取的提示词如下，上面三个SubAgent也是参考这份提示词让豆包写出来的：开始AI4AI准备好program\.md与SubAgent后，别忘了进行git init，一切就绪后就可以开始运行了，以下内容是我实操的输入，供大家参考： 嗨，看看program\.md，让我们开始一个新的实验！让我们先完成前置。 这个时候Agent会先简单分析你的项目文件，了解你定义任务program\.md，并新建了git分支。我提前准备好了一个baseline model不用他浪费时间重跑一遍，并把记录填到了result\.tsv 目前已经完成所有前置。我接近24h没有休息了，终于完成了目前的baseline model\(v0，已存在于auto\_dir/results\.tsv\)，我需要一段长时间的深度休息，请你充分合理使用subagent功能进行新实验，做到无人值守优化模型\(不是写脚本run\_autoresearch\.sh，用于持续执行实验；而是你来一步一步进行探索与优化\)，希望我回来后有一个好消息 跑了大概6、7个小时后，其中大部分时间其实是等，因为流程跑通了，也有一点好结果，我就中断了 暂停所有工作，中断正在运行的训练脚本，汇总工作 结果迭代了11个版本，在有限的训练周期类，提高了AUPRC10个点，感觉还是很不错的在这个过程中，总共使用了56个SubAgent\+1个Main Agent，但是有点不太好的点是，除了task\_awaiter，其他Agent都是codex自带的，并没用我写的。为什么会用这么多个SubAgent呢，是因为并不是一个SubAgent从头用到尾的，主Agent下发的任务完成后，就会把SubAgent关闭。由于SubAgent的引入，免不得Token的消耗，不过我用的是中转（这里不做推荐），用起来也不心疼Token usage: total=1,972,501 input=1,926,236 \(\+ 14,491,008 cached\) output=46,265 \(reasoning 13,199\)踩过的坑 用的是linux系统下的docker，codex指令/experimental可以选择一个什么沙盒的选项，如果选了会没办法多卡执行torchrun，报错有关端口权限问题，不过codex v0\.115\.0版本/experimental就没有沙盒相关的选项了 autoresearch作者发现codex似乎对never stop相关的提示词不敏感\(Codex doesn’t seem to work? · Issue \#57 · karpathy/autoresearch\)，导致容易过早中断，一开始做时也发现了，我没有采用很高明的策略，只对program\.md和对话进行优化，上面的内容已经优化过了 如果没有强调使用SubAgent的话，还是会自己埋头干，同样地，我也只对program\.md和对话进行优化，上面的内容已经优化过了 尾记 原本是想发在人类之光上的，但是想了想这帖子内容其实也没那么科研学术，也只是对Agent技术的尝试 在写贴的时候，香港大学数据智能实验室（HKUDS）也发布了ClawTeam : Agent Swarm Intelligence \( GitHub \- HKUDS/ClawTeam: ClawTeam: Agent Swarm Intelligence \(One Command → Full Automation\) · GitHub \)。做得内容比我更加高级，不过与autoresearch一样，主要在训练配置上进行优化。HKUDS非常高产，十分佩服，同时也希望社区会有更多的优秀Agent技术与项目分享。

\(3\.19记\)上述Agent帮我改的模型在有限的训练时间内有显著提升，于是我改回正常训练后，模型性能却并没有比我自己的baseline model好，缺乏后续上升的动力。我也看了一下模型究竟做了什么改动，虽然我在program\.md中允许开放性大改模型，但是其实添加的模块都是小东西，质量也一般（我觉得冗余）。我猜测是：

program\.md中"优先选择更简洁的方案" 模型训练数据并没有很新，并不清楚当前SOTA架构 模型架构与常见的不太同，模型不熟悉，难以找到很好的切入点

\(3\.24记\)最近一项工作也使用了autoresearch的思路去优化模型，做的工作是预测蛋白熔点\[ The Melting Point \- by Frank Gao \- Dimension Research; DimensionCap/autoresearch\_thermo: Autoresearch Framework For Protein Thermostability\]，其中采用多种优化方案，包括贝叶斯优化\(Bayesian optimization, BO\)、 约束性Agent\(Restricted Agent, RA\)、开放性Agent\(Unleashed Agent, UA\)以及组合优化，结果如下图。我上面的实验对应的是UA，结论和我有相同：“UA比较开放，虽然会有意想不到的技术加进来，但是并不是优选（会改差甚至改错）”。Agent在超参调优的能力还是高于架构调整的，配合上贝叶斯优化，可以有效缩短试错空间

阅读时间

5 分钟

上次访问

由 duduke 于 3月 18 日 发布

由 aitech 于 3月 18 日 发布

由 Devon 于 3月 18 日 发布

由 fredjim1225 于 3月 19 日 发布

由 Rosec 于 3月 19 日 发布

由 futures 于 3月 22 日 发布

由 ctf101 于 3月 23 日 发布

由 Duzc24 于 3月 23 日 发布

由 PuQing 于 3月 23 日 发布

由 leohong 于 3月 23 日 发布

---
*由 GitHub MD Saver 插件自动保存*
