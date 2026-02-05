# astrbot_plugin_BizCanvas
随手写的自用兼容画图模型


#插件和以往插件重复，不再通过官方发布，
#以下内容ai生成

# AstrBot Plugin Biz Canvas

基于 Gemini 模型的高级生图插件，专为 AstrBot 设计。支持多协议接口调用、自动地址清洗及网络连接优化。

## 功能特性

* **智能生图**: 支持文生图 (Draw) 与图生图 (Edit)，自动适配 OpenAI Chat 与 Google Responses 协议。
* **风格预设**: 内置手办化 (Figurine)、真人化 (Cosplay)、宝可梦化 (Pokemon) 三种专属风格转换指令。
* **网络优化**: 自动清洗网关 URL 路径，集成浏览器 UA 伪装与 SSL 优化，有效解决 CDN (如 Cloudflare/EdgeOne) 524/554 超时问题。
* **智能解析**: 增强型图片提取逻辑，支持从非标准 JSON 字段、Markdown 及 HTML 中提取图片链接。

## 配置说明

* **gateway_url**: API 网关地址。支持填写 IP 或域名，插件会自动修正重复的 /v1 后缀。
* **api_route_type**: API 路由格式。标准 NewAPI/OneAPI 请选择 `chat_completions`。
* **model_name**: 模型名称，建议使用 `gemini-2.0-flash-exp` 或 `gemini-3-pro`。
* **image_config**: 可调整生成图片的画幅比例 (Ratio) 和分辨率 (Size)。

## 使用方法

### 1. 基础绘图 (LLM Tool)
在对话中直接发送自然语言需求，由 LLM 自动调用工具：
* "画一只在雨中漫步的猫"
* "把这张图改成赛博朋克风格" (需引用或附带图片)

### 2. 风格化指令 (正则触发)
发送图片时附带以下关键词（支持直接发送或引用图片）：
* **手办化 [描述]**: 将 2D 图片转换为高质量 3D 手办风格。
* **cos化 [描述]**: 将动漫角色转换为真人 Cosplay 风格。
* **宝可梦 [描述]**: 将图片重绘为宝可梦图鉴风格。
