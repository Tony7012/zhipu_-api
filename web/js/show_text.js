// 在节点下方显示模型返回的文本结果(执行后更新)
// 参考 pysssss/ComfyUI-Custom-Scripts 的 ShowText 实现(MIT)
import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

const NODE_NAMES = ["ZhipuImageAnalysis", "ZhipuTextChat"];

app.registerExtension({
    name: "ZhipuAI.ShowText",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (!NODE_NAMES.includes(nodeData?.name)) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);

            // 创建只读的多行展示区(不参与工作流序列化)
            const { widget } = ComfyWidgets.STRING(
                this,
                "show_text",
                ["", { default: "", multiline: true }],
                app
            );
            widget.inputEl.readOnly = true;
            widget.inputEl.style.opacity = 0.8;
            widget.serialize = false;

            return r;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            const text = message?.text?.[0];
            if (text == null) return;
            const widget = this.widgets?.find((w) => w.name === "show_text");
            if (widget) {
                widget.value = text;
                if (this.onResize) {
                    this.onResize(this.size);
                }
                app.graph.setDirtyCanvas(true, false);
            }
        };
    },
});
