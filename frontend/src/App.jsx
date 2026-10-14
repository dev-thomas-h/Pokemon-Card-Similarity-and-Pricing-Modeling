import React, { useState, useEffect } from 'react';
import { 
  Search, Filter, Sparkles, TrendingUp, Layers, 
  Calendar, Award, ArrowUpRight, ShieldCheck, RefreshCw, Zap
} from 'lucide-react';

const FALLBACK_IMAGE = 'https://assets.tcgdex.net/univ/base/base1/symbol';

export default function App() {
  const [activeTab, setActiveTab] = useState('explorer');
  const [options, setOptions] = useState({ sets: [], rarities: [] });
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedSet, setSelectedSet] = useState('all');
  const [selectedRarity, setSelectedRarity] = useState('all');
  const [searchResults, setSearchResults] = useState([]);
  const [selectedCardId, setSelectedCardId] = useState('sv03.5-199'); // Default: 151 Charizard ex
  const [similarityData, setSimilarityData] = useState(null);
  const [topGems, setTopGems] = useState([]);
  
  const [loadingSearch, setLoadingSearch] = useState(false);
  const [loadingCard, setLoadingCard] = useState(false);
  const [loadingGems, setLoadingGems] = useState(false);

  // Load dropdown options on mount
  useEffect(() => {
    fetch('/api/options')
      .then(res => res.json())
      .then(data => {
        setOptions(data);
      })
      .catch(err => console.error('Failed to load filter options:', err));
  }, []);

  // Search modern cards whenever query, set, or rarity changes
  useEffect(() => {
    setLoadingSearch(true);
    const delayDebounce = setTimeout(() => {
      const params = new URLSearchParams();
      if (searchQuery) params.append('q', searchQuery);
      if (selectedSet !== 'all') params.append('set_id', selectedSet);
      if (selectedRarity !== 'all') params.append('rarity', selectedRarity);
      params.append('limit', '20');

      fetch(`/api/cards/search?${params.toString()}`)
        .then(res => res.json())
        .then(data => {
          setSearchResults(data.results || []);
          setLoadingSearch(false);
        })
        .catch(err => {
          console.error('Search error:', err);
          setLoadingSearch(false);
        });
    }, 250);

    return () => clearTimeout(delayDebounce);
  }, [searchQuery, selectedSet, selectedRarity]);

  // Load similarity comps whenever selectedCardId changes
  useEffect(() => {
    if (!selectedCardId) return;
    setLoadingCard(true);
    fetch(`/api/similarity/${selectedCardId}?k=5`)
      .then(res => {
        if (!res.ok) throw new Error('Card not found');
        return res.json();
      })
      .then(data => {
        setSimilarityData(data);
        setLoadingCard(false);
      })
      .catch(err => {
        console.error('Similarity load error:', err);
        setLoadingCard(false);
      });
  }, [selectedCardId]);

  // Load top gems when scanner tab is opened or filters change
  const loadTopGems = () => {
    setLoadingGems(true);
    const params = new URLSearchParams();
    if (selectedRarity !== 'all') params.append('rarity', selectedRarity);
    if (selectedSet !== 'all') params.append('set_id', selectedSet);
    params.append('min_price', '5.0');
    params.append('limit', '15');

    fetch(`/api/top-gems?${params.toString()}`)
      .then(res => res.json())
      .then(data => {
        setTopGems(data.gems || []);
        setLoadingGems(false);
      })
      .catch(err => {
        console.error('Failed to load top gems:', err);
        setLoadingGems(false);
      });
  };

  useEffect(() => {
    if (activeTab === 'gems') {
      loadTopGems();
    }
  }, [activeTab, selectedSet, selectedRarity]);

  const target = similarityData?.target_card;
  const projections = similarityData?.projections;
  const vintageNeighbors = similarityData?.vintage_neighbors || [];

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="brand">
          <div className="brand-icon"></div>
          <div>
            <h1 className="brand-title">PokéMaturation Engine</h1>
            <p className="brand-subtitle">Machine Learning Appraisal & Vintage Precedent Predictor</p>
          </div>
        </div>

        <nav className="header-nav">
          <button 
            className={`nav-tab ${activeTab === 'explorer' ? 'active' : ''}`}
            onClick={() => setActiveTab('explorer')}
          >
            <Sparkles size={16} />
            Card Matcher
          </button>
          <button 
            className={`nav-tab ${activeTab === 'gems' ? 'active' : ''}`}
            onClick={() => setActiveTab('gems')}
          >
            <TrendingUp size={16} />
            Top Gems Scanner
          </button>
        </nav>
      </header>

      {/* Global Filter & Search Panel */}
      <section className="search-panel">
        <div className="search-grid">
          <div className="input-group">
            <Search className="input-icon" size={18} />
            <input 
              type="text" 
              className="search-input"
              placeholder="Search modern cards (e.g. Charizard, Gengar, sv03.5-199)..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>

          <div className="input-group">
            <Layers className="input-icon" size={18} />
            <select 
              className="select-input"
              value={selectedSet}
              onChange={(e) => setSelectedSet(e.target.value)}
            >
              <option value="all">All Modern Sets (2016+)</option>
              {options.sets.map(s => (
                <option key={s.id} value={s.id}>
                  {s.name} {s.year ? `(${s.year})` : ''}
                </option>
              ))}
            </select>
          </div>

          <div className="input-group">
            <Award className="input-icon" size={18} />
            <select 
              className="select-input"
              value={selectedRarity}
              onChange={(e) => setSelectedRarity(e.target.value)}
            >
              <option value="all">All Rarities</option>
              {options.rarities.map(r => (
                <option key={r} value={r}>{r}</option>
              ))}
            </select>
          </div>

          <button 
            className="inspect-btn"
            style={{ height: '44px', padding: '0 18px', display: 'flex', alignItems: 'center', gap: '8px' }}
            onClick={() => {
              setSearchQuery('');
              setSelectedSet('all');
              setSelectedRarity('all');
            }}
          >
            <RefreshCw size={14} />
            Reset
          </button>
        </div>

        {/* Quick Search Results Tray */}
        {searchResults.length > 0 && (
          <div className="search-results-tray">
            <div className="results-label">Matching Modern Cards ({searchResults.length}):</div>
            <div className="chips-scroller">
              {searchResults.map(c => (
                <div 
                  key={c.card_id} 
                  className={`card-chip ${c.card_id === selectedCardId ? 'active' : ''}`}
                  onClick={() => {
                    setSelectedCardId(c.card_id);
                    if (activeTab !== 'explorer') setActiveTab('explorer');
                  }}
                >
                  <img 
                    src={c.image_url} 
                    alt={c.name}
                    className="chip-img"
                    onError={(e) => { e.target.src = FALLBACK_IMAGE; }}
                  />
                  <div className="chip-info">
                    <span className="chip-name">{c.name}</span>
                    <span className="chip-sub">
                      {c.set_name} • {c.market_price ? `$${c.market_price.toFixed(2)}` : 'N/A'}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </section>

      {/* VIEW 1: Card Matcher / Appraisal */}
      {activeTab === 'explorer' && (
        <>
          {loadingCard ? (
            <div className="loading-box">
              <div className="spinner"></div>
              <p>Analyzing collector dimensions and retrieving vintage comps...</p>
            </div>
          ) : target ? (
            <>
              <div className="appraisal-layout">
                {/* Target Modern Card Card */}
                <div className={`target-card-box ${target.has_holo ? 'holo-sheen' : ''}`}>
                  <img 
                    src={target.image_url} 
                    alt={target.name}
                    className="card-img-large"
                    onError={(e) => { e.target.src = FALLBACK_IMAGE; }}
                  />
                  <h2 className="card-name-title">{target.name}</h2>
                  <p className="card-set-title">
                    {target.set_name} ({target.release_year}) • #{target.card_id}
                  </p>

                  <div className="badge-row">
                    <span className="badge badge-gold">Prestige Tier {target.rarity_tier}/5</span>
                    <span className="badge badge-purple">{target.rarity}</span>
                    {target.is_secret_rare && <span className="badge badge-cyan">Secret Rare</span>}
                    {target.has_holo && <span className="badge badge-gold">Holo Finish</span>}
                    {target.illustrator && <span className="badge badge-gray">{target.illustrator}</span>}
                  </div>

                  <div style={{ marginTop: 'auto', textAlign: 'center', width: '100%' }}>
                    <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 700 }}>
                      Current Modern Market Price
                    </span>
                    <div style={{ fontSize: '1.8rem', fontWeight: 900, color: '#ffffff' }}>
                      {target.current_market_price ? `$${target.current_market_price.toFixed(2)}` : 'Unpriced'}
                    </div>
                  </div>
                </div>

                {/* Valuation Projection Showcase */}
                <div className="valuation-banner">
                  <div>
                    <div className="valuation-header">
                      <div>
                        <div className="val-tag">
                          <Zap size={13} />
                          Empirical Maturation Appraisal
                        </div>
                        <h3 style={{ fontSize: '1.45rem', fontWeight: 800 }}>Vintage Counterpart Projection</h3>
                        <p style={{ color: 'var(--text-muted)', fontSize: '0.88rem' }}>
                          Price equilibrium of vintage cards (`&lt; 2016`) sharing this card's prestige profile.
                        </p>
                      </div>

                      {projections?.upside_multiplier && (
                        <div className="stat-upside-badge" style={{ fontSize: '1.1rem', padding: '6px 14px' }}>
                          <ArrowUpRight size={18} style={{ display: 'inline', verticalAlign: '-3px' }} />
                          {projections.upside_multiplier}x Upside
                        </div>
                      )}
                    </div>

                    <div className="stat-showcase">
                      <div className="stat-item">
                        <span className="stat-lbl">Projected Median Value</span>
                        <div className="stat-val-proj">
                          ${projections?.vintage_median_price?.toFixed(2)}
                        </div>
                      </div>

                      <div className="stat-item">
                        <span className="stat-lbl">Average Vintage Price</span>
                        <div style={{ fontSize: '1.6rem', fontWeight: 800, color: '#f1f5f9' }}>
                          ${projections?.vintage_mean_price?.toFixed(2)}
                        </div>
                      </div>

                      <div className="stat-item">
                        <span className="stat-lbl">Historical Precedent Range</span>
                        <div className="stat-range">
                          ${projections?.vintage_min_price?.toFixed(2)} – ${projections?.vintage_max_price?.toFixed(2)}
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="valuation-footer-note">
                    <ShieldCheck size={20} color="#10b981" />
                    <span>
                      Grounded across <strong>{projections?.sample_size} historical benchmark cards</strong> with matching rarity, holo finish, and subject prestige.
                    </span>
                  </div>
                </div>
              </div>

              {/* Vintage Comps Grid */}
              <section>
                <h3 className="section-headline">
                  <Award size={20} color="var(--accent-gold)" />
                  Top Historical Vintage Counterparts (&lt; 2016)
                </h3>

                <div className="vintage-grid">
                  {vintageNeighbors.map((v, i) => (
                    <div key={v.card_id} className="vintage-card">
                      <span className="similarity-pill">
                        {v.similarity_score}% Match
                      </span>
                      <img 
                        src={v.image_url} 
                        alt={v.name}
                        className="v-card-img"
                        onError={(e) => { e.target.src = FALLBACK_IMAGE; }}
                      />
                      <h4 className="v-card-name">{v.name}</h4>
                      <p className="v-card-set">
                        {v.set_name} ({v.release_year})
                      </p>
                      <span className="badge badge-purple" style={{ marginBottom: '10px' }}>
                        {v.rarity}
                      </span>
                      <div className="v-card-price">
                        ${v.market_price ? v.market_price.toFixed(2) : 'N/A'}
                      </div>
                    </div>
                  ))}
                </div>
              </section>
            </>
          ) : (
            <div className="loading-box">
              <p>No card selected. Use the search bar above to look up any modern card!</p>
            </div>
          )}
        </>
      )}

      {/* VIEW 2: Top Gems Scanner */}
      {activeTab === 'gems' && (
        <section>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
            <div>
              <h2 style={{ fontSize: '1.4rem', fontWeight: 800 }}>💎 Highest Maturation Potential Gems</h2>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.88rem' }}>
                Modern cards with the highest historical price upside compared to vintage counterparts.
              </p>
            </div>
            <button className="inspect-btn" onClick={loadTopGems}>
              <RefreshCw size={14} style={{ marginRight: '6px', display: 'inline' }} />
              Refresh Scan
            </button>
          </div>

          {loadingGems ? (
            <div className="loading-box">
              <div className="spinner"></div>
              <p>Scanning modern sets against vintage benchmarks...</p>
            </div>
          ) : topGems.length > 0 ? (
            <div className="gems-table-wrapper">
              <table className="gems-table">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Card Name</th>
                    <th>Set / Era</th>
                    <th>Rarity</th>
                    <th>Current Price</th>
                    <th>Projected Vintage Value</th>
                    <th>Upside Multiplier</th>
                    <th>Action</th>
                  </tr>
                </thead>
                <tbody>
                  {topGems.map((g, idx) => (
                    <tr key={g.card_id}>
                      <td style={{ fontWeight: 800, color: 'var(--text-muted)' }}>{idx + 1}</td>
                      <td>
                        <div className="gem-card-cell">
                          <img 
                            src={g.image_url} 
                            alt={g.name} 
                            className="gem-mini-img"
                            onError={(e) => { e.target.src = FALLBACK_IMAGE; }}
                          />
                          <div>
                            <div style={{ fontWeight: 700 }}>{g.name}</div>
                            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>#{g.card_id}</div>
                          </div>
                        </div>
                      </td>
                      <td>
                        <div>{g.set_name}</div>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{g.release_year}</div>
                      </td>
                      <td>
                        <span className="badge badge-purple">{g.rarity}</span>
                      </td>
                      <td style={{ fontWeight: 700 }}>
                        ${g.current_price.toFixed(2)}
                      </td>
                      <td style={{ fontWeight: 800, color: 'var(--accent-gold)' }}>
                        ${g.vintage_median_price.toFixed(2)}
                      </td>
                      <td>
                        <span className="stat-upside-badge">
                          {g.upside_multiplier}x
                        </span>
                      </td>
                      <td>
                        <button 
                          className="inspect-btn"
                          onClick={() => {
                            setSelectedCardId(g.card_id);
                            setActiveTab('explorer');
                          }}
                        >
                          Inspect Comps
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="loading-box">
              <p>No cards matched the current gem scan filters.</p>
            </div>
          )}
        </section>
      )}
    </div>
  );
}
