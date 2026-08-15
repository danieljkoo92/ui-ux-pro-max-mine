# ui-ux-pro-max-mine

Personal trimmed fork of [nextlevelbuilder/ui-ux-pro-max-skill](https://github.com/nextlevelbuilder/ui-ux-pro-max-skill), cut down to the parts that serve software development.

## What this fork changes

Upstream ships **7 skills**. Every one of their descriptions loads into the model's context on every session, whether used or not. This fork keeps **3**:

| Skill | Kept? | Why |
|---|---|---|
| `ui-ux-pro-max` | ✅ | The core engine — searchable local database (styles, palettes, font pairings, UX guidelines, charts, 22 stacks) queried by `search.py` |
| `ui-styling` | ✅ | shadcn/ui + Radix + Tailwind implementation, dark mode, accessible components |
| `design-system` | ✅ | Three-layer token architecture (primitive → semantic → component) |
| `design` | ❌ | Logo generation, corporate identity programs, social photos — agency/marketing scope, and depends on Gemini API keys |
| `banner-design` | ❌ | Social/ad banner creative — marketing scope |
| `brand` | ❌ | Brand voice, messaging frameworks, style guides — marketing scope |
| `slides` | ❌ | HTML presentation generation |

Also removed from the tree: `cli/` (npm distribution), `src/` (upstream's build source-of-truth), `stack/` (examples — the skill's own stack data lives under `.claude/skills/ui-ux-pro-max/data/stacks/`), plus screenshots, gallery, and CI config. Repo goes from ~21 MB to ~9.4 MB.

`.claude/skills/ui-ux-pro-max/scripts/tests/` was dropped because it reads from `src/`, which no longer exists here.

## Known upstream issue carried over

`design-system/scripts/generate-slide.py` references `../../../assets/design-tokens.css`, which resolves to `.claude/assets/design-tokens.css`. **That directory does not exist upstream either** — slide generation from that script is already broken in the original. Unfixed here since slide generation isn't the reason this fork exists.

## Usage

The search tool is stdlib-only Python 3:

```bash
python "${CLAUDE_PLUGIN_ROOT}/.claude/skills/ui-ux-pro-max/scripts/search.py" "<query>" --domain <domain>
```

Domains: `style`, `product`, `color`, `typography`, `ux`, `chart`, `gsap`, `icon`, plus per-stack data.

## Re-syncing with upstream

This fork is a snapshot, not a live mirror — upstream improvements do not arrive automatically. To pull them in:

```bash
git remote add upstream https://github.com/nextlevelbuilder/ui-ux-pro-max-skill.git
git fetch upstream
git diff HEAD upstream/main -- .claude/skills/ui-ux-pro-max .claude/skills/ui-styling .claude/skills/design-system
```

Merge selectively — a straight merge would reintroduce the four removed skills.

## License

MIT, inherited from upstream. Original copyright belongs to nextlevelbuilder; see [LICENSE](LICENSE). All design data, scripts, and skill content are upstream's work. This fork only removes files and rewrites the manifests.
