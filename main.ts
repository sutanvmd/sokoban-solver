// Sokoban solver: push-based A* with per-goal push-distance tables, an exact
// matching heuristic, static dead-square and 2x2 freeze pruning, and a room
// decomposition (generated levels are independent rooms off a hallway).
// Components split by articulation points are merged back into balanced
// regions (equal boxes and goals), each region is solved on its own, and
// the walks between regions are stitched together. All search budgets are
// deterministic expansion counters, never the clock, so two runs of this
// program produce byte-identical output.
//
// Memory discipline (the grader gives 512 MB): search nodes live in flat
// typed arrays (packed box positions, parent index, the single push that
// created the node) instead of per-node objects holding move strings; the
// full move string is rebuilt once at the end by replaying the pushes.
// The visited set is a Map keyed by a 32-bit numeric hash whose stored
// value packs a second independent 32-bit hash together with g, so a key
// collision can never silently prune a distinct state. Every structure is
// hard-capped: past the cap a search stops and reports unsolved for that
// region instead of growing until the process aborts.
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

// ---------- state hashing (two independent 32-bit hashes) ----------

function hashA(boxes, base, nb, rep) {
  let x = (rep + 0x9e3779b9) | 0;
  for (let i = 0; i < nb; i++) {
    x = Math.imul(x ^ (boxes[base + i] + 0x85ebca6b), 0xc2b2ae35);
    x ^= x >>> 13;
  }
  x = Math.imul(x ^ (x >>> 16), 0x85ebca6b);
  x ^= x >>> 13;
  return x >>> 0;
}

function hashB(boxes, base, nb, rep) {
  let x = (rep * 31 + 0x27d4eb2f) | 0;
  for (let i = nb - 1; i >= 0; i--) {
    x = Math.imul(x ^ (boxes[base + i] + 0x165667b1), 0x27d4eb2d);
    x ^= x >>> 15;
  }
  x = Math.imul(x ^ (x >>> 15), 0x2c1b3c6d);
  x ^= x >>> 12;
  return x >>> 0;
}

// ---------- heuristic: box->goal matching on push distances ----------

// dists[g][cell] is the push distance from cell to goal g. `dp` is a
// caller-owned scratch buffer of size 2^nb, reused across calls.
function heuristic(dists, ng, boxes, base, nb, dp) {
  if (nb === 0) return 0;
  for (let i = 0; i < nb; i++) {
    const b = boxes[base + i];
    let any = false;
    for (let g = 0; g < ng; g++) if (dists[g][b] >= 0) any = true;
    if (!any) return Infinity;
  }
  if (nb <= 10 && ng === nb) {
    const size = 1 << nb;
    dp.fill(Infinity, 0, size);
    dp[0] = 0;
    for (let mask = 0; mask < size; mask++) {
      let i = 0;
      for (let m = mask; m; m &= m - 1) i++;
      if (i >= nb || dp[mask] === Infinity) continue;
      const b = boxes[base + i];
      for (let g = 0; g < nb; g++) {
        if (mask & (1 << g)) continue;
        const c = dists[g][b];
        if (c < 0) continue;
        const nm = mask | (1 << g);
        if (dp[mask] + c < dp[nm]) dp[nm] = dp[mask] + c;
      }
    }
    return dp[size - 1];
  }
  let sum = 0;
  for (let i = 0; i < nb; i++) {
    const b = boxes[base + i];
    let best = Infinity;
    for (let g = 0; g < ng; g++) {
      const c = dists[g][b];
      if (c >= 0 && c < best) best = c;
    }
    sum += best;
  }
  return sum;
}

// ---------- push-based A* over a region ----------

const VISITED_MAX = 320000; // visited entries; past this the search stops
const NODES_MAX = 1000000; // stored nodes; past this the search stops
const GPACK = 1048576; // 2^20: g values always stay far below this

// Boxes must stay inside `allowed`; `hardBlocked` cells (other regions'
// boxes) are permanent obstacles for the player. Returns
// { moves, pushes, playerEnd } or null when unsolved within `cap` expansions.
function solveRegion(L, allowed, boxAllowed, boxes0, goals, player0, hardBlocked, cap) {
  const { n, nbr, wall } = L;
  const nb = boxes0.length;
  const ng = goals.length;
  const dists = pushDistances(L, allowed, goals);
  const dead = new Uint8Array(n);
  for (let c = 0; c < n; c++) {
    if (!allowed[c] || L.goal[c]) continue;
    let alive = false;
    for (const dist of dists) if (dist[c] >= 0) alive = true;
    if (!alive) dead[c] = 1;
  }
  for (const b of boxes0) if (dead[b]) return null;
  const allOnGoal = (arr, base) => {
    for (let i = 0; i < nb; i++) if (!L.goal[arr[base + i]]) return false;
    return true;
  };
  const dp = nb <= 10 && ng === nb ? new Float64Array(1 << nb) : null;
  const hOf = (arr, base) =>
    heuristic(dists, ng, arr, base, nb, dp ?? new Float64Array(0));
  if (allOnGoal(boxes0, 0)) return { moves: '', pushes: 0, playerEnd: player0 };
  const h0 = hOf(boxes0, 0);
  if (!isFinite(h0)) return null;

  const blocked = new Uint8Array(n);
  const seen = new Int32Array(n);
  const par = new Int32Array(n);
  const inBox = new Uint8Array(n);
  let stamp = 0;

  // Flat node store, grown by doubling. Per node: nb packed box cells,
  // parent index, player cell after the push, the push direction, g, h, f.
  let capNodes = 65536;
  let boxArr = new Int32Array(capNodes * nb);
  let parentArr = new Int32Array(capNodes);
  let playerArr = new Int32Array(capNodes);
  let gArr = new Int32Array(capNodes);
  let hArr = new Int32Array(capNodes);
  let fArr = new Int32Array(capNodes);
  let dirArr = new Int8Array(capNodes);
  let nodeCount = 0;
  const grow = () => {
    const nc = capNodes * 2;
    const nb2 = new Int32Array(nc * nb);
    nb2.set(boxArr);
    boxArr = nb2;
    const growI = (a) => {
      const b = new Int32Array(nc);
      b.set(a);
      return b;
    };
    parentArr = growI(parentArr);
    playerArr = growI(playerArr);
    gArr = growI(gArr);
    hArr = growI(hArr);
    fArr = growI(fArr);
    const nd = new Int8Array(nc);
    nd.set(dirArr);
    dirArr = nd;
    capNodes = nc;
  };
  const addNode = (srcArr, srcBase, player, parent, dir, g, h) => {
    if (nodeCount >= NODES_MAX) return -1;
    if (nodeCount === capNodes) grow();
    const id = nodeCount++;
    const base = id * nb;
    for (let i = 0; i < nb; i++) boxArr[base + i] = srcArr[srcBase + i];
    playerArr[id] = player;
    parentArr[id] = parent;
    dirArr[id] = dir;
    gArr[id] = g;
    hArr[id] = h;
    fArr[id] = g + h;
    return id;
  };

  const heap = [];
  const less = (a, b) => {
    if (fArr[a] !== fArr[b]) return fArr[a] < fArr[b];
    if (hArr[a] !== hArr[b]) return hArr[a] < hArr[b];
    return a < b;
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

  const root = addNode(boxes0, 0, player0, -1, -1, 0, h0);
  pushHeap(root);
  const visited = new Map();
  let expansions = 0;

  const frozen2x2 = (moved) => {
    const mx = moved % L.w;
    const my = (moved / L.w) | 0;
    for (let oy = -1; oy <= 0; oy++) {
      for (let ox = -1; ox <= 0; ox++) {
        let ok = true;
        let allSolid = true;
        let badBox = false;
        for (let dy = 0; dy < 2 && ok; dy++) {
          for (let dx = 0; dx < 2; dx++) {
            const x = mx + ox + dx;
            const y = my + oy + dy;
            if (x < 0 || y < 0 || x >= L.w || y >= L.h) {
              ok = false;
              break;
            }
            const c = y * L.w + x;
            if (wall[c]) continue;
            if (inBox[c]) {
              if (!L.goal[c]) badBox = true;
              continue;
            }
            allSolid = false;
          }
        }
        if (ok && allSolid && badBox) return true;
      }
    }
    return false;
  };

  // Replay the pushes from the root to `goalId` and rebuild the full move
  // string (walks between pushes recomputed by flood fill, exactly as the
  // search saw them).
  const reconstruct = (goalId) => {
    const chain = [];
    for (let cur = goalId; parentArr[cur] >= 0; cur = parentArr[cur]) chain.push(cur);
    chain.reverse();
    const curBoxes = boxes0.slice();
    let player = player0;
    let moves = '';
    for (const id of chain) {
      const d = dirArr[id];
      const boxFrom = playerArr[id];
      const stand = nbr[boxFrom * 4 + (d ^ 1)];
      const dest = nbr[boxFrom * 4 + d];
      for (let c = 0; c < n; c++) blocked[c] = wall[c] || hardBlocked[c] ? 1 : 0;
      for (const b of curBoxes) blocked[b] = 1;
      stamp++;
      flood(L, blocked, player, stamp, seen, par);
      moves += walkString(L, par, player, stand) + DIRS[d].ch.toUpperCase();
      curBoxes[curBoxes.indexOf(boxFrom)] = dest;
      player = boxFrom;
    }
    return { moves, pushes: gArr[goalId], playerEnd: playerArr[goalId] };
  };

  const childBoxes = new Int32Array(nb);
  while (heap.length) {
    if (++expansions > cap) return null;
    const id = popHeap();
    const base = id * nb;
    const g = gArr[id];
    for (let c = 0; c < n; c++) blocked[c] = wall[c] || hardBlocked[c] ? 1 : 0;
    for (let i = 0; i < nb; i++) blocked[boxArr[base + i]] = 1;
    stamp++;
    const rep = flood(L, blocked, playerArr[id], stamp, seen, par);
    const kA = hashA(boxArr, base, nb, rep);
    const kB = hashB(boxArr, base, nb, rep);
    const prev = visited.get(kA);
    if (prev !== undefined && Math.floor(prev / GPACK) === kB && prev % GPACK <= g) continue;
    if (visited.size >= VISITED_MAX) return null;
    visited.set(kA, kB * GPACK + g);
    if (allOnGoal(boxArr, base)) return reconstruct(id);
    for (let i = 0; i < nb; i++) inBox[boxArr[base + i]] = 1;
    for (let bi = 0; bi < nb; bi++) {
      const b = boxArr[base + bi];
      for (let d = 0; d < 4; d++) {
        const stand = nbr[b * 4 + (d ^ 1)];
        const dest = nbr[b * 4 + d];
        if (stand < 0 || dest < 0) continue;
        if (seen[stand] !== stamp) continue;
        if (!boxAllowed[dest] || wall[dest] || inBox[dest]) continue;
        if (dead[dest]) continue;
        for (let i = 0; i < nb; i++) childBoxes[i] = boxArr[base + i];
        childBoxes[bi] = dest;
        // insertion sort the single moved box into place
        let j = bi;
        while (j > 0 && childBoxes[j - 1] > childBoxes[j]) {
          const t = childBoxes[j - 1];
          childBoxes[j - 1] = childBoxes[j];
          childBoxes[j] = t;
          j--;
        }
        while (j < nb - 1 && childBoxes[j + 1] < childBoxes[j]) {
          const t = childBoxes[j + 1];
          childBoxes[j + 1] = childBoxes[j];
          childBoxes[j] = t;
          j++;
        }
        inBox[b] = 0;
        inBox[dest] = 1;
        const frozen = frozen2x2(dest);
        inBox[dest] = 0;
        inBox[b] = 1;
        if (frozen) continue;
        const h = hOf(childBoxes, 0);
        if (!isFinite(h)) continue;
        const cid = addNode(childBoxes, 0, b, id, d, g + 1, h);
        if (cid < 0) return null;
        pushHeap(cid);
      }
    }
    for (let i = 0; i < nb; i++) inBox[boxArr[base + i]] = 0;
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
    const res = solveRegion(L, allowed, boxAllowed, rBoxes, rGoals, start, hardBlocked, 120000);
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
    200000,
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
