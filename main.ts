// Sokoban solver: push-based A* with per-goal push-distance tables, an exact
// matching heuristic, static dead-square and 2x2 freeze pruning, and a room
// decomposition (generated levels are independent rooms off a hallway).
// Components split by articulation points are merged back into balanced
// regions (equal boxes and goals), each region is solved on its own, and
// the walks between regions are stitched together. All search budgets are
// deterministic expansion counters, never the clock, so two runs of this
// program produce byte-identical output.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const IN = process.env.MAKEFUN_INPUT_DIR ?? process.env.SOKO_IN ?? '/input';
const OUT = process.env.MAKEFUN_OUTPUT_DIR ?? process.env.SOKO_OUT ?? '/output';

const DIRS = [
  { dx: 0, dy: -1, ch: 'u' },
  { dx: 0, dy: 1, ch: 'd' },
  { dx: -1, dy: 0, ch: 'l' },
  { dx: 1, dy: 0, ch: 'r' },
];

// ---------- level parsing ----------

function parseLevel(text) {
  const rows = text.replace(/\r/g, '').split('\n');
  while (rows.length && rows[rows.length - 1] === '') rows.pop();
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const n = w * h;
  const wall = new Uint8Array(n).fill(1);
  const goal = new Uint8Array(n);
  const boxes = [];
  let player = -1;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const c = rows[y][x] ?? '#';
      const i = y * w + x;
      if (c === '#') continue;
      wall[i] = 0;
      if (c === '.' || c === '*' || c === '+') goal[i] = 1;
      if (c === '$' || c === '*') boxes.push(i);
      if (c === '@' || c === '+') player = i;
    }
  }
  const nbr = new Int32Array(n * 4).fill(-1);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      for (let d = 0; d < 4; d++) {
        const nx = x + DIRS[d].dx;
        const ny = y + DIRS[d].dy;
        if (nx >= 0 && ny >= 0 && nx < w && ny < h) nbr[i * 4 + d] = ny * w + nx;
      }
    }
  }
  const goals = [];
  for (let i = 0; i < n; i++) if (goal[i]) goals.push(i);
  boxes.sort((a, b) => a - b);
  return { w, h, n, wall, goal, goals, boxes, player, nbr };
}

// ---------- push-distance tables over an allowed cell set ----------

function pushDistances(L, allowed, goals) {
  const { n, nbr } = L;
  return goals.map((g) => {
    const dist = new Int32Array(n).fill(-1);
    dist[g] = 0;
    const q = [g];
    for (let qi = 0; qi < q.length; qi++) {
      const c = q[qi];
      for (let d = 0; d < 4; d++) {
        const prev = nbr[c * 4 + (d ^ 1)];
        if (prev < 0 || !allowed[prev]) continue;
        const stand = nbr[prev * 4 + (d ^ 1)];
        if (stand < 0 || !allowed[stand]) continue;
        if (dist[prev] === -1) {
          dist[prev] = dist[c] + 1;
          q.push(prev);
        }
      }
    }
    return dist;
  });
}

// ---------- player flood fill ----------

function flood(L, blocked, start, stamp, seen, par) {
  const { nbr } = L;
  const q = [start];
  seen[start] = stamp;
  par[start] = -1;
  let rep = start;
  for (let qi = 0; qi < q.length; qi++) {
    const c = q[qi];
    if (c < rep) rep = c;
    for (let d = 0; d < 4; d++) {
      const nx = nbr[c * 4 + d];
      if (nx >= 0 && seen[nx] !== stamp && !blocked[nx]) {
        seen[nx] = stamp;
        par[nx] = c;
        q.push(nx);
      }
    }
  }
  return rep;
}

function walkString(L, par, from, to) {
  if (from === to) return '';
  const chars = [];
  let c = to;
  while (c !== from) {
    const p = par[c];
    for (let d = 0; d < 4; d++) if (L.nbr[p * 4 + d] === c) chars.push(DIRS[d].ch);
    c = p;
  }
  return chars.reverse().join('');
}

// ---------- heuristic: box->goal matching on push distances ----------

function heuristic(dists, goals, boxes) {
  const n = boxes.length;
  if (n === 0) return 0;
  const cost = boxes.map((b) => dists.map((dist) => dist[b]));
  for (let i = 0; i < n; i++) {
    let any = false;
    for (let g = 0; g < goals.length; g++) if (cost[i][g] >= 0) any = true;
    if (!any) return Infinity;
  }
  if (n <= 10 && goals.length === n) {
    const size = 1 << n;
    const dp = new Float64Array(size).fill(Infinity);
    dp[0] = 0;
    for (let mask = 0; mask < size; mask++) {
      let i = 0;
      for (let m = mask; m; m &= m - 1) i++;
      if (i >= n || dp[mask] === Infinity) continue;
      for (let g = 0; g < n; g++) {
        if (mask & (1 << g)) continue;
        const c = cost[i][g];
        if (c < 0) continue;
        const nm = mask | (1 << g);
        if (dp[mask] + c < dp[nm]) dp[nm] = dp[mask] + c;
      }
    }
    return dp[size - 1];
  }
  let sum = 0;
  for (let i = 0; i < n; i++) {
    let best = Infinity;
    for (let g = 0; g < goals.length; g++) {
      const c = cost[i][g];
      if (c >= 0 && c < best) best = c;
    }
    sum += best;
  }
  return sum;
}

// ---------- push-based A* over a region ----------

// Boxes must stay inside `allowed`; `hardBlocked` cells (other regions'
// boxes) are permanent obstacles for the player. Returns
// { moves, pushes, playerEnd } or null when unsolved within `cap` expansions.
function solveRegion(L, allowed, boxAllowed, boxes0, goals, player0, hardBlocked, cap) {
  const { n, nbr, wall } = L;
  const dists = pushDistances(L, allowed, goals);
  const dead = new Uint8Array(n);
  for (let c = 0; c < n; c++) {
    if (!allowed[c] || L.goal[c]) continue;
    let alive = false;
    for (const dist of dists) if (dist[c] >= 0) alive = true;
    if (!alive) dead[c] = 1;
  }
  for (const b of boxes0) if (dead[b]) return null;
  const goalSet = new Set(goals);
  const isGoalBoxes = (bx) => bx.every((b) => goalSet.has(b));
  if (isGoalBoxes(boxes0)) return { moves: '', pushes: 0, playerEnd: player0 };

  const blocked = new Uint8Array(n);
  const seen = new Int32Array(n);
  const par = new Int32Array(n);
  let stamp = 0;

  const h0 = heuristic(dists, goals, boxes0);
  if (!isFinite(h0)) return null;

  const nodes = [];
  const heap = [];
  const less = (a, b) => {
    const na = nodes[a];
    const nb = nodes[b];
    if (na.f !== nb.f) return na.f < nb.f;
    if (na.h !== nb.h) return na.h < nb.h;
    return na.seq < nb.seq;
  };
  const pushHeap = (id) => {
    heap.push(id);
    let i = heap.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i], heap[p])) break;
      [heap[i], heap[p]] = [heap[p], heap[i]];
      i = p;
    }
  };
  const popHeap = () => {
    const top = heap[0];
    const last = heap.pop();
    if (heap.length) {
      heap[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1;
        const r = l + 1;
        let m = i;
        if (l < heap.length && less(heap[l], heap[m])) m = l;
        if (r < heap.length && less(heap[r], heap[m])) m = r;
        if (m === i) break;
        [heap[i], heap[m]] = [heap[m], heap[i]];
        i = m;
      }
    }
    return top;
  };

  let seq = 0;
  nodes.push({
    boxes: boxes0.slice(),
    player: player0,
    g: 0,
    h: h0,
    f: h0,
    parent: -1,
    seg: '',
    seq: seq++,
  });
  pushHeap(0);
  const visited = new Map();
  let expansions = 0;

  const frozen2x2 = (boxSet, moved) => {
    const mx = moved % L.w;
    const my = (moved / L.w) | 0;
    for (let oy = -1; oy <= 0; oy++) {
      for (let ox = -1; ox <= 0; ox++) {
        const cells = [];
        let ok = true;
        for (let dy = 0; dy < 2 && ok; dy++) {
          for (let dx = 0; dx < 2; dx++) {
            const x = mx + ox + dx;
            const y = my + oy + dy;
            if (x < 0 || y < 0 || x >= L.w || y >= L.h) {
              ok = false;
              break;
            }
            cells.push(y * L.w + x);
          }
        }
        if (!ok) continue;
        let allSolid = true;
        let badBox = false;
        for (const c of cells) {
          if (wall[c]) continue;
          if (boxSet.has(c)) {
            if (!goalSet.has(c)) badBox = true;
            continue;
          }
          allSolid = false;
        }
        if (allSolid && badBox) return true;
      }
    }
    return false;
  };

  while (heap.length) {
    if (++expansions > cap) return null;
    const id = popHeap();
    const node = nodes[id];
    for (let c = 0; c < n; c++) blocked[c] = wall[c] || hardBlocked[c] ? 1 : 0;
    for (const b of node.boxes) blocked[b] = 1;
    stamp++;
    const rep = flood(L, blocked, node.player, stamp, seen, par);
    const key = node.boxes.join(',') + '|' + rep;
    if (visited.has(key) && visited.get(key) <= node.g) continue;
    visited.set(key, node.g);
    if (isGoalBoxes(node.boxes)) {
      const segs = [];
      for (let cur = node; cur.parent >= 0; cur = nodes[cur.parent]) segs.push(cur.seg);
      return {
        moves: segs.reverse().join(''),
        pushes: node.g,
        playerEnd: node.player,
      };
    }
    const boxSet = new Set(node.boxes);
    for (const b of node.boxes) {
      for (let d = 0; d < 4; d++) {
        const stand = nbr[b * 4 + (d ^ 1)];
        const dest = nbr[b * 4 + d];
        if (stand < 0 || dest < 0) continue;
        if (seen[stand] !== stamp) continue;
        if (!boxAllowed[dest] || wall[dest] || boxSet.has(dest)) continue;
        if (dead[dest]) continue;
        const nbx = node.boxes.slice();
        nbx[nbx.indexOf(b)] = dest;
        nbx.sort((a, b2) => a - b2);
        const nSet = new Set(nbx);
        if (frozen2x2(nSet, dest)) continue;
        const h = heuristic(dists, goals, nbx);
        if (!isFinite(h)) continue;
        const seg = walkString(L, par, node.player, stand) + DIRS[d].ch.toUpperCase();
        nodes.push({
          boxes: nbx,
          player: b,
          g: node.g + 1,
          h,
          f: node.g + 1 + h,
          parent: id,
          seg,
          seq: seq++,
        });
        pushHeap(nodes.length - 1);
      }
    }
  }
  return null;
}

// ---------- articulation points and balanced regions ----------

function articulationPoints(L) {
  const { n, nbr, wall } = L;
  const disc = new Int32Array(n).fill(-1);
  const low = new Int32Array(n);
  const isAP = new Uint8Array(n);
  let time = 0;
  const dfs = (u, parent) => {
    disc[u] = low[u] = time++;
    let children = 0;
    for (let d = 0; d < 4; d++) {
      const v = nbr[u * 4 + d];
      if (v < 0 || wall[v]) continue;
      if (disc[v] === -1) {
        children++;
        dfs(v, u);
        low[u] = Math.min(low[u], low[v]);
        if (parent !== -1 && low[v] >= disc[u]) isAP[u] = 1;
      } else if (v !== parent) {
        low[u] = Math.min(low[u], disc[v]);
      }
    }
    if (parent === -1 && children > 1) isAP[u] = 1;
  };
  for (let i = 0; i < n; i++) if (!wall[i] && disc[i] === -1) dfs(i, -1);
  return isAP;
}

function walkPath(L, boxSet, from, to) {
  if (from === to) return '';
  const { n, nbr, wall } = L;
  const prev = new Int32Array(n).fill(-2);
  prev[from] = -1;
  const q = [from];
  for (let qi = 0; qi < q.length; qi++) {
    const c = q[qi];
    if (c === to) break;
    for (let d = 0; d < 4; d++) {
      const nx = nbr[c * 4 + d];
      if (nx >= 0 && prev[nx] === -2 && !wall[nx] && !boxSet.has(nx)) {
        prev[nx] = c;
        q.push(nx);
      }
    }
  }
  if (prev[to] === -2) return null;
  const chars = [];
  for (let c = to; c !== from; c = prev[c]) {
    const p = prev[c];
    for (let d = 0; d < 4; d++) if (nbr[p * 4 + d] === c) chars.push(DIRS[d].ch);
  }
  return chars.reverse().join('');
}

// Split the floor into units (components after removing articulation
// points, plus each articulation cell as its own unit), then merge
// adjacent units until every unit holds as many boxes as goals. Returns
// the units that hold boxes, as arrays of cells, or null.
function balancedRegions(L) {
  const { n, nbr, wall } = L;
  const isAP = articulationPoints(L);
  const unitOf = new Int32Array(n).fill(-1);
  let nUnit = 0;
  for (let i = 0; i < n; i++) {
    if (wall[i] || isAP[i] || unitOf[i] !== -1) continue;
    const q = [i];
    unitOf[i] = nUnit;
    for (let qi = 0; qi < q.length; qi++) {
      const c = q[qi];
      for (let d = 0; d < 4; d++) {
        const nx = nbr[c * 4 + d];
        if (nx >= 0 && !wall[nx] && !isAP[nx] && unitOf[nx] === -1) {
          unitOf[nx] = nUnit;
          q.push(nx);
        }
      }
    }
    nUnit++;
  }
  for (let i = 0; i < n; i++) if (!wall[i] && isAP[i]) unitOf[i] = nUnit++;
  if (nUnit === 0) return null;

  const parent = Array.from({ length: nUnit }, (_, i) => i);
  const find = (a) => {
    while (parent[a] !== a) {
      parent[a] = parent[parent[a]];
      a = parent[a];
    }
    return a;
  };
  const cellsOf = Array.from({ length: nUnit }, () => []);
  for (let i = 0; i < n; i++) if (unitOf[i] >= 0) cellsOf[unitOf[i]].push(i);
  const boxesIn = (root) =>
    cellsOf[root].reduce((s, c) => s + (L.boxes.includes(c) ? 1 : 0), 0);
  const goalsIn = (root) => cellsOf[root].reduce((s, c) => s + L.goal[c], 0);
  const adjacent = (root) => {
    const set = new Set();
    for (const c of cellsOf[root]) {
      for (let d = 0; d < 4; d++) {
        const nx = nbr[c * 4 + d];
        if (nx >= 0 && unitOf[nx] >= 0) {
          const r = find(unitOf[nx]);
          if (r !== root) set.add(r);
        }
      }
    }
    return set;
  };

  for (let guard = 0; guard < nUnit * 2; guard++) {
    const roots = [...new Set(cellsOf.map((_, i) => find(i)))];
    const bad = roots.filter((r) => boxesIn(r) !== goalsIn(r));
    if (bad.length === 0) break;
    if (roots.length === 1) return null; // cannot balance: counts differ
    bad.sort((a, b) => cellsOf[a].length - cellsOf[b].length);
    const r = bad[0];
    const imb = boxesIn(r) - goalsIn(r);
    let best = -1;
    let bestScore = Infinity;
    for (const o of adjacent(r)) {
      const merged = Math.abs(imb + (boxesIn(o) - goalsIn(o)));
      const score = merged * 100000 + cellsOf[o].length;
      if (score < bestScore) {
        bestScore = score;
        best = o;
      }
    }
    if (best < 0) return null;
    parent[find(best)] = find(r);
    cellsOf[find(r)] = cellsOf[find(r)].concat(cellsOf[best]);
    cellsOf[best] = [];
  }
  const regions = [];
  const seenRoots = new Set();
  for (let i = 0; i < nUnit; i++) {
    const r = find(i);
    if (seenRoots.has(r)) continue;
    seenRoots.add(r);
    if (boxesIn(r) !== goalsIn(r)) return null;
    if (boxesIn(r) > 0) regions.push(cellsOf[r]);
  }
  return regions.length ? regions : null;
}

function solveByRegions(L) {
  const regions = balancedRegions(L);
  if (!regions) return null;
  regions.sort(
    (a, b) =>
      Math.min(...a.filter((c) => L.boxes.includes(c))) -
      Math.min(...b.filter((c) => L.boxes.includes(c))),
  );
  const boxSet = new Set(L.boxes);
  let player = L.player;
  let moves = '';
  for (const cells of regions) {
    const cellSet = new Set(cells);
    const rBoxes = cells.filter((c) => boxSet.has(c)).sort((a, b) => a - b);
    const rGoals = cells.filter((c) => L.goal[c]);
    const allowed = new Uint8Array(L.n);
    for (const c of cells) allowed[c] = 1;
    // doors: floor cells outside the region adjacent to it; the player may
    // also stand there while pushing at the region's edge
    for (const c of cells) {
      for (let d = 0; d < 4; d++) {
        const nx = L.nbr[c * 4 + d];
        if (nx >= 0 && !L.wall[nx] && !cellSet.has(nx)) allowed[nx] = 1;
      }
    }
    // boxes of this region may not leave the region's own cells
    const boxAllowed = new Uint8Array(L.n);
    for (const c of cells) boxAllowed[c] = 1;
    const hardBlocked = new Uint8Array(L.n);
    for (const b of boxSet) if (!cellSet.has(b)) hardBlocked[b] = 1;
    let start = -1;
    let walk = '';
    if (cellSet.has(player)) {
      start = player;
    } else {
      // walk to any boundary cell of the region, then step in handled by
      // the region search starting from that boundary cell
      let bestW = null;
      for (const c of cells) {
        const w = walkPath(L, boxSet, player, c);
        if (w !== null && (bestW === null || w.length < bestW.length)) {
          bestW = w;
          start = c;
        }
      }
      if (start < 0) return null;
      walk = bestW;
    }
    const res = solveRegion(L, allowed, boxAllowed, rBoxes, rGoals, start, hardBlocked, 250000);
    if (!res) return null;
    moves += walk + res.moves;
    for (const b of rBoxes) boxSet.delete(b);
    for (const g of rGoals) boxSet.add(g);
    player = res.playerEnd;
  }
  return moves;
}

// ---------- top-level solve ----------

function solveLevel(text) {
  const L = parseLevel(text);
  if (L.boxes.every((b) => L.goal[b])) return '';
  const byRegions = solveByRegions(L);
  if (byRegions !== null) return byRegions;
  const allowed = new Uint8Array(L.n);
  for (let i = 0; i < L.n; i++) allowed[i] = L.wall[i] ? 0 : 1;
  const res = solveRegion(
    L,
    allowed,
    allowed,
    L.boxes.slice(),
    L.goals,
    L.player,
    new Uint8Array(L.n),
    400000,
  );
  return res ? res.moves : '';
}

for (const id of readdirSync(IN).sort()) {
  let moves = '';
  try {
    moves = solveLevel(readFileSync(join(IN, id, 'level.txt'), 'utf8'));
  } catch {
    moves = '';
  }
  mkdirSync(join(OUT, id), { recursive: true });
  writeFileSync(join(OUT, id, 'solution.txt'), `${moves}\n`);
}
