# SNF 静态前端 — 部署指南

这是一个**完全静态**的网页:浏览器加载 `models.json`(已烤好的模型系数),
全部预测都在本地 JS 跑,**没有任何后端调用**。

部署到任何静态托管(GitHub Pages / Netlify / Vercel / 自家 nginx)都行。

---

## A. 本地预览(几秒钟)

```bash
cd static_app
python3 -m http.server 9876
# 浏览器打开 http://localhost:9876
```

或者:

```bash
npx serve static_app -l 9876
```

---

## B. 部署到 GitHub Pages(推荐 · 免费 · 自动)

仓库根目录已经有 `.github/workflows/deploy-static.yml`,合并到 `main` 后会自动部署。
**第一次启用**只需在 GitHub UI 里做一次设置:

1. **打开仓库** → Settings → Pages
2. **Source** 选 **GitHub Actions**(不是 "Deploy from a branch")
3. **保存** → 然后到 **Actions** 标签等待 "Deploy static_app to GitHub Pages" workflow 跑完(~30 秒)
4. 完成后,在 `Actions` 页面顶部会看到部署的 URL,也可以在 Settings → Pages 看到。
   通常是 `https://<你的用户名>.github.io/<仓库名>/`

> 例如本仓库就是 `https://liuxx839.github.io/snf_analysis/`。

### 之后怎么更新

1. 改了模型(系数变化)?
   ```bash
   python3 src/export_static_models.py    # 重新导出 models.json
   git add static_app/models.json
   git commit -m "update model coefficients"
   git push
   ```
2. 改了前端(`index.html / style.css / app.js`)?直接 commit + push 即可。

每次推送只要动了 `static_app/` 下的文件,GitHub Actions 就会自动重新部署。

---

## C. 部署到 Netlify / Vercel(更快,1 分钟)

### Netlify
1. 登录 [app.netlify.com](https://app.netlify.com) → Add new site → Import from Git
2. 选这个仓库,build command 留空,**publish directory 填 `static_app`**
3. 点 Deploy。完成后 Netlify 会给一个 `xxx.netlify.app` 链接

### Vercel
1. 登录 [vercel.com](https://vercel.com) → New Project → 选仓库
2. **Root Directory** 改成 `static_app`,Framework Preset 选 "Other"
3. Deploy

---

## D. 自家服务器 / 任何 nginx

把 `static_app/` 整个目录复制到 web root 就行,不需要 Node / Python。

```nginx
location /snf/ {
  alias /var/www/static_app/;
  index index.html;
}
```

---

## 文件清单

```
static_app/
├── index.html       # UI
├── style.css        # 样式
├── app.js           # 推理逻辑(纯 JS)
├── models.json      # 烤好的模型系数 (~370 KB)
└── README.md        # 本文件
```

---

## 模型怎么重新烤?

```bash
# 在仓库根目录
python3 src/export_static_models.py
# → 写到 static_app/models.json
```

如果训练数据换了 / 默认特征集变了 / 想加新算法,改完 `src/training.py` 或
`src/survival.py` 后重新跑这一行就行。
