import { useEffect, useMemo, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  ArrowRight,
  Check,
  ChefHat,
  Clock,
  Home,
  Image as ImageIcon,
  MapPin,
  RefreshCw,
  Search,
  ShoppingBasket,
  Sparkles,
  Store,
  Utensils,
} from "lucide-react";

const API_BASE = "http://127.0.0.1:8000";

const STAGES = {
  LANDING: "landing",
  ANALYZING: "analyzing",
  INGREDIENTS: "ingredients",
  RECIPES: "recipes",
};

const analysisSteps = [
  "Scanning fridge image",
  "Detecting ingredients",
  "Preparing recipe suggestions",
];

function imageUrl(path) {
  if (!path) return "";
  return `${API_BASE}${path}`;
}

function formatTime(minutes) {
  if (minutes === null || minutes === undefined) return "Time not listed";
  return `${minutes} min`;
}

function getShopBadgeClass(type) {
  return type === "Budget" ? "badge budget" : "badge selection";
}

function App() {
  const [stage, setStage] = useState(STAGES.LANDING);
  const [images, setImages] = useState([]);
  const [selectedImageId, setSelectedImageId] = useState("");
  const [selectedImage, setSelectedImage] = useState(null);
  const [detectedIngredients, setDetectedIngredients] = useState([]);
  const [recipes, setRecipes] = useState([]);
  const [shops, setShops] = useState({});
  const [selectedArea, setSelectedArea] = useState("");
  const [selectedShopIndex, setSelectedShopIndex] = useState(0);
  const [preference, setPreference] = useState("all");
  const [recipeCount, setRecipeCount] = useState(5);
  const [activeInstruction, setActiveInstruction] = useState(null);
  const [loadingImages, setLoadingImages] = useState(true);
  const [error, setError] = useState("");

  const selectedShop = useMemo(() => {
    if (!selectedArea || !shops[selectedArea]) return null;
    return shops[selectedArea][selectedShopIndex] || shops[selectedArea][0];
  }, [shops, selectedArea, selectedShopIndex]);

  useEffect(() => {
    loadInitialData();
  }, []);

  async function loadInitialData() {
    try {
      setLoadingImages(true);
      setError("");

      const [imagesRes, shopsRes] = await Promise.all([
        fetch(`${API_BASE}/api/images`),
        fetch(`${API_BASE}/api/shops`),
      ]);

      if (!imagesRes.ok) {
        throw new Error("Could not load available fridge images.");
      }

      if (!shopsRes.ok) {
        throw new Error("Could not load shop suggestions.");
      }

      const imagesData = await imagesRes.json();
      const shopsData = await shopsRes.json();

      setImages(imagesData.images || []);

      if (imagesData.images && imagesData.images.length > 0) {
        setSelectedImageId(imagesData.images[0].image_id);
      }

      setShops(shopsData.areas || {});

      const firstArea = Object.keys(shopsData.areas || {})[0];
      if (firstArea) {
        setSelectedArea(firstArea);
      }
    } catch (err) {
      setError(err.message || "Something went wrong while loading the app.");
    } finally {
      setLoadingImages(false);
    }
  }

  async function analyzeSelectedImage() {
    if (!selectedImageId) return;

    setStage(STAGES.ANALYZING);
    setError("");

    try {
      await wait(2200);

      const res = await fetch(`${API_BASE}/api/images/${encodeURIComponent(selectedImageId)}`);

      if (!res.ok) {
        throw new Error("Could not analyze this image.");
      }

      const data = await res.json();

      setSelectedImage(data);
      setDetectedIngredients(data.detected_ingredients || []);
      setStage(STAGES.INGREDIENTS);
    } catch (err) {
      setError(err.message || "Could not analyze this image.");
      setStage(STAGES.LANDING);
    }
  }

  async function findRecipes() {
    if (!selectedImageId) return;

    try {
      setError("");

      const res = await fetch(`${API_BASE}/api/recipes`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          image_id: selectedImageId,
          top_n: 12,
          preference,
        }),
      });

      if (!res.ok) {
        throw new Error("Could not find recipes for this image.");
      }

      const data = await res.json();

      setRecipes(data.recipes || []);
      setStage(STAGES.RECIPES);
    } catch (err) {
      setError(err.message || "Could not find recipes.");
    }
  }

  function resetApp() {
    setStage(STAGES.LANDING);
    setSelectedImage(null);
    setDetectedIngredients([]);
    setRecipes([]);
    setPreference("all");
    setRecipeCount(5);
    setActiveInstruction(null);
  }

  const selectedImagePreview = images.find((img) => img.image_id === selectedImageId);

  return (
    <div className="app-shell">
      <div className="bg-orb orb-one" />
      <div className="bg-orb orb-two" />

      <main className="app-container">
        <TopBar onReset={resetApp} />

        {error && (
          <motion.div
            className="error-banner"
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
          >
            {error}
          </motion.div>
        )}

        <StepIndicator stage={stage} />

        <AnimatePresence mode="wait">
          {stage === STAGES.LANDING && (
            <LandingScreen
              key="landing"
              images={images}
              selectedImageId={selectedImageId}
              setSelectedImageId={setSelectedImageId}
              selectedImagePreview={selectedImagePreview}
              loadingImages={loadingImages}
              onAnalyze={analyzeSelectedImage}
            />
          )}

          {stage === STAGES.ANALYZING && (
            <AnalyzingScreen key="analyzing" />
          )}

          {stage === STAGES.INGREDIENTS && (
            <IngredientsScreen
              key="ingredients"
              selectedImage={selectedImage}
              detectedIngredients={detectedIngredients}
              preference={preference}
              setPreference={setPreference}
              onFindRecipes={findRecipes}
              onChooseAnother={resetApp}
            />
          )}

          {stage === STAGES.RECIPES && (
            <RecipesScreen
              key="recipes"
              recipes={recipes}
              recipeCount={recipeCount}
              setRecipeCount={setRecipeCount}
              selectedArea={selectedArea}
              setSelectedArea={setSelectedArea}
              shops={shops}
              selectedShopIndex={selectedShopIndex}
              setSelectedShopIndex={setSelectedShopIndex}
              selectedShop={selectedShop}
              activeInstruction={activeInstruction}
              setActiveInstruction={setActiveInstruction}
              onChooseAnother={resetApp}
            />
          )}
        </AnimatePresence>
      </main>
    </div>
  );
}

function TopBar({ onReset }) {
  return (
    <header className="topbar">
      <button className="brand-button" onClick={onReset}>
        <span className="brand-icon">
          <ChefHat size={20} />
        </span>
        <span>Fridge-to-Recipe</span>
      </button>
    </header>
  );
}

function StepIndicator({ stage }) {
  const steps = [
    { id: STAGES.LANDING, label: "Select image" },
    { id: STAGES.ANALYZING, label: "Analyze" },
    { id: STAGES.INGREDIENTS, label: "Ingredients" },
    { id: STAGES.RECIPES, label: "Recipes" },
  ];

  const activeIndex = steps.findIndex((step) => step.id === stage);

  return (
    <div className="stepper">
      {steps.map((step, index) => {
        const isActive = index === activeIndex;
        const isDone = index < activeIndex;

        return (
          <div className="stepper-item" key={step.id}>
            <div className={`step-dot ${isActive ? "active" : ""} ${isDone ? "done" : ""}`}>
              {isDone ? <Check size={15} /> : index + 1}
            </div>
            <span className={isActive ? "step-label active" : "step-label"}>
              {step.label}
            </span>
            {index < steps.length - 1 && <div className="step-line" />}
          </div>
        );
      })}
    </div>
  );
}

function LandingScreen({
  images,
  selectedImageId,
  setSelectedImageId,
  selectedImagePreview,
  loadingImages,
  onAnalyze,
}) {
  const [hoverMode, setHoverMode] = useState(null);

  return (
    <motion.section
      className="hero-grid"
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -16 }}
      transition={{ duration: 0.35 }}
    >
      <div className="hero-copy">
        <div className="eyebrow">
          <Sparkles size={16} />
          From fridge photo to dinner plan
        </div>

        <h1>See what’s inside. Find what to cook.</h1>

        <p className="hero-text">
          Select a fridge image and get recipe ideas based on the visible ingredients,
          with nearby shopping suggestions for anything still needed.
        </p>

        <div className="select-panel">
          <label>Select a fridge image</label>

          {loadingImages ? (
            <div className="loading-box">Loading images...</div>
          ) : (
            <select
              value={selectedImageId}
              onChange={(event) => setSelectedImageId(event.target.value)}
            >
              {images.map((image) => (
                <option key={image.image_id} value={image.image_id}>
                  {image.filename}
                </option>
              ))}
            </select>
          )}

          <button
            className="primary-button"
            onClick={onAnalyze}
            disabled={!selectedImageId || loadingImages}
          >
            Analyze fridge
            <ArrowRight size={18} />
          </button>
        </div>

        <div className="feature-row">
          <FeatureCard icon={<ImageIcon size={20} />} title="Detect ingredients" />
          <FeatureCard icon={<Utensils size={20} />} title="Recommend recipes" />
          <FeatureCard icon={<ShoppingBasket size={20} />} title="Plan missing items" />
        </div>
      </div>

      <div className="hero-preview">
        <div className={`preview-frame ${hoverMode ? "preview-active" : ""}`}>
          {selectedImagePreview ? (
            <img
              src={imageUrl(selectedImagePreview.image_url)}
              alt="Selected fridge"
            />
          ) : (
            <div className="empty-preview">No image selected</div>
          )}

          <AnimatePresence>
            {hoverMode === "ingredients" && (
              <motion.div
                className="image-overlay ingredient-overlay"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                <span className="ingredient-zone zone-one" />
                <span className="ingredient-zone zone-two" />
                <span className="ingredient-zone zone-three" />
                <span className="ingredient-zone zone-four" />
              </motion.div>
            )}

            {hoverMode === "shops" && (
              <motion.div
                className="image-overlay shop-overlay"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                <div className="shop-preview-panel">
                  <div className="shop-preview-title">Nearby shopping options</div>

                  <div className="shop-brand-strip">
                    <span>REWE</span>
                    <span>EDEKA</span>
                    <span>ALDI</span>
                    <span>Lidl</span>
                    <span>PENNY</span>
                    <span>Netto</span>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        <motion.div
          className="floating-card floating-card-one"
          onMouseEnter={() => setHoverMode("ingredients")}
          onMouseLeave={() => setHoverMode(null)}
          initial={{ opacity: 0, x: 20, y: 20 }}
          animate={{ opacity: 1, x: 0, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <Search size={18} />
          <div>
            <strong>Visible ingredients</strong>
            <span>hover to highlight</span>
          </div>
        </motion.div>

        <motion.div
          className="floating-card floating-card-two"
          onMouseEnter={() => setHoverMode("shops")}
          onMouseLeave={() => setHoverMode(null)}
          initial={{ opacity: 0, x: -20, y: -20 }}
          animate={{ opacity: 1, x: 0, y: 0 }}
          transition={{ delay: 0.35 }}
        >
          <Store size={18} />
          <div>
            <strong>Local shops</strong>
            <span>hover to preview</span>
          </div>
        </motion.div>
      </div>
    </motion.section>
  );
}

function FeatureCard({ icon, title }) {
  return (
    <div className="feature-card">
      <span>{icon}</span>
      <p>{title}</p>
    </div>
  );
}

function AnalyzingScreen() {
  const [activeStep, setActiveStep] = useState(0);

  useEffect(() => {
    const timers = analysisSteps.map((_, index) =>
      setTimeout(() => setActiveStep(index), index * 700)
    );

    return () => timers.forEach(clearTimeout);
  }, []);

  return (
    <motion.section
      className="center-stage"
      initial={{ opacity: 0, scale: 0.97 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.97 }}
      transition={{ duration: 0.35 }}
    >
      <div className="analysis-card">
        <motion.div
          className="scanner-icon"
          animate={{ rotate: [0, 8, -8, 0], scale: [1, 1.05, 1] }}
          transition={{ repeat: Infinity, duration: 1.4 }}
        >
          <Sparkles size={34} />
        </motion.div>

        <h2>Analyzing your fridge</h2>
        <p>The assistant is preparing recipe ideas from the selected image.</p>

        <div className="analysis-steps">
          {analysisSteps.map((step, index) => (
            <motion.div
              className={`analysis-step ${index <= activeStep ? "active" : ""}`}
              key={step}
              initial={{ opacity: 0.35 }}
              animate={{ opacity: index <= activeStep ? 1 : 0.35 }}
            >
              <span className="analysis-dot">
                {index < activeStep ? <Check size={14} /> : index + 1}
              </span>
              {step}
            </motion.div>
          ))}
        </div>

        <div className="progress-track">
          <motion.div
            className="progress-fill"
            initial={{ width: "0%" }}
            animate={{ width: "100%" }}
            transition={{ duration: 2.1, ease: "easeInOut" }}
          />
        </div>
      </div>
    </motion.section>
  );
}

function IngredientsScreen({
  selectedImage,
  detectedIngredients,
  preference,
  setPreference,
  onFindRecipes,
  onChooseAnother,
}) {
  return (
    <motion.section
      className="content-grid"
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -16 }}
      transition={{ duration: 0.35 }}
    >
      <div className="image-card">
        <div className="card-header">
          <div>
            <span className="eyebrow compact">Selected fridge</span>
            <h2>Analyzed image</h2>
          </div>
        </div>

        {selectedImage?.image_url ? (
          <img
            src={imageUrl(selectedImage.image_url)}
            alt="Analyzed fridge"
          />
        ) : (
          <div className="empty-preview">Image unavailable</div>
        )}
      </div>

      <div className="details-card">
        <span className="eyebrow compact">Detected from image</span>
        <h2>Visible ingredients</h2>

        <div className="chips">
          {detectedIngredients.length > 0 ? (
            detectedIngredients.map((ingredient) => (
              <span className="chip" key={ingredient}>
                {ingredient}
              </span>
            ))
          ) : (
            <div className="empty-state">
              No clear ingredients were detected for this image.
            </div>
          )}
        </div>

        <div className="recipe-controls">
          <label>Recipe preference</label>
          <div className="segmented">
            <button
              className={preference === "all" ? "active" : ""}
              onClick={() => setPreference("all")}
            >
              All
            </button>
            <button
              className={preference === "vegetarian" ? "active" : ""}
              onClick={() => setPreference("vegetarian")}
            >
              Vegetarian
            </button>
            <button
              className={preference === "quick" ? "active" : ""}
              onClick={() => setPreference("quick")}
            >
              Quick
            </button>
          </div>
        </div>

        <div className="button-row">
          <button
            className="primary-button"
            onClick={onFindRecipes}
            disabled={detectedIngredients.length === 0}
          >
            Find recipes
            <ArrowRight size={18} />
          </button>

          <button className="ghost-button" onClick={onChooseAnother}>
            <RefreshCw size={17} />
            Choose another
          </button>
        </div>
      </div>
    </motion.section>
  );
}

function RecipesScreen({
  recipes,
  recipeCount,
  setRecipeCount,
  selectedArea,
  setSelectedArea,
  shops,
  selectedShopIndex,
  setSelectedShopIndex,
  selectedShop,
  activeInstruction,
  setActiveInstruction,
  onChooseAnother,
}) {
  const areas = Object.keys(shops);
  const currentShops = selectedArea ? shops[selectedArea] || [] : [];

  useEffect(() => {
    setSelectedShopIndex(0);
  }, [selectedArea, setSelectedShopIndex]);

  return (
    <motion.section
      className="recipes-layout"
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -16 }}
      transition={{ duration: 0.35 }}
    >
      <div className="recipes-top-grid">
        <div className="recipes-title-block">
          <span className="eyebrow compact">Recipe suggestions</span>
          <h1>Recipes you can make</h1>

          <div className="recipe-top-actions">
            <div className="recipe-count-select">
              <label>Number of recipes</label>
              <select
                value={recipeCount}
                onChange={(event) => setRecipeCount(Number(event.target.value))}
              >
                {[3, 5, 8, 10].map((count) => (
                  <option key={count} value={count}>
                    {count} recipes
                  </option>
                ))}
              </select>
            </div>

            <button className="ghost-button" onClick={onChooseAnother}>
              <Home size={17} />
              New image
            </button>
          </div>
        </div>

        <div className="shop-panel shop-panel-compact">
          <div className="shop-panel-title">
            <MapPin size={19} />
            Nearby shopping
          </div>

          <div className="shop-grid compact-shop-grid">
            <div>
              <label>Select your area</label>
              <select
                value={selectedArea}
                onChange={(event) => setSelectedArea(event.target.value)}
              >
                {areas.map((area) => (
                  <option key={area} value={area}>
                    {area}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label>Select a shop</label>
              <select
                value={selectedShopIndex}
                onChange={(event) => setSelectedShopIndex(Number(event.target.value))}
              >
                {currentShops.map((shop, index) => (
                  <option key={`${shop.name}-${shop.address}`} value={index}>
                    {shop.name} — {shop.address}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {selectedShop && (
            <div className="selected-shop compact-selected-shop">
              <div>
                <strong>{selectedShop.name}</strong>
                <span>{selectedShop.address}</span>
              </div>
              <span className={getShopBadgeClass(selectedShop.type)}>
                {selectedShop.type}
              </span>
            </div>
          )}
        </div>
      </div>

      {recipes.length === 0 ? (
        <div className="empty-state large">
          No recipes found for this image.
        </div>
      ) : (
        <div className="recipe-list">
          {recipes.slice(0, recipeCount).map((recipe, index) => (
            <RecipeCard
              key={`${recipe.title}-${index}`}
              recipe={recipe}
              index={index}
              selectedShop={selectedShop}
              isOpen={activeInstruction === index}
              onToggle={() =>
                setActiveInstruction(activeInstruction === index ? null : index)
              }
            />
          ))}
        </div>
      )}

      <p className="disclaimer">
        Store suggestions are static local guidance. Opening hours, prices, and availability may vary.
      </p>
    </motion.section>
  );
}

function RecipeCard({ recipe, index, selectedShop, isOpen, onToggle }) {
  const difficultyLabel = getDifficultyLabel(recipe.missing_difficulty);
  const difficultyClass = getDifficultyClass(recipe.missing_difficulty);

  const matchedIngredients = recipe.matched_ingredients || [];
  const missingIngredients = recipe.missing_ingredients || [];
  const grocerySuggestions = recipe.grocery_suggestions || [];
  const instructions = recipe.instructions || [];

  const missingPreview = missingIngredients.slice(0, 4).join(", ");
  const primaryGrocerySuggestion = grocerySuggestions[0] || null;

  return (
    <motion.article
      className="recipe-card"
      initial={{ opacity: 0, y: 18 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.06 }}
    >
      <div className="recipe-top">
        <div>
          <h2>{recipe.title}</h2>

          <div className="recipe-meta">
            {recipe.cuisine &&
              recipe.cuisine.toLowerCase() !== "unknown" &&
              recipe.cuisine.toLowerCase() !== "cuisine not listed" && (
                <span>
                  <Utensils size={15} />
                  {recipe.cuisine}
                </span>
              )}

            {recipe.meal_type &&
              recipe.meal_type.toLowerCase() !== "unknown" &&
              recipe.meal_type.toLowerCase() !== "meal type not listed" && (
                <span>
                  <ChefHat size={15} />
                  {recipe.meal_type}
                </span>
              )}

            {recipe.prep_time !== null && recipe.prep_time !== undefined && (
              <span>
                <Clock size={15} />
                {formatTime(recipe.prep_time)}
              </span>
            )}
          </div>
        </div>

        <div className="recipe-score-stack">
          <div className="match-score">
            <strong>{recipe.match_percentage}%</strong>
            <span>match</span>
          </div>

          <div className={`difficulty-badge ${difficultyClass}`}>
            {difficultyLabel}
          </div>
        </div>
      </div>

      <div className="ingredient-columns">
        <div>
          <h3>Matched ingredients</h3>

          <div className="chips compact-chips">
            {matchedIngredients.length > 0 ? (
              matchedIngredients.map((item) => (
                <span className="chip success" key={item}>
                  {item}
                </span>
              ))
            ) : (
              <span className="empty-text">None</span>
            )}
          </div>
        </div>

        <div>
          <h3>Still needed</h3>

          <div className="chips compact-chips">
            {missingIngredients.length > 0 ? (
              missingIngredients.map((item) => (
                <span className="chip warning" key={item}>
                  {item}
                </span>
              ))
            ) : (
              <span className="empty-text">Nothing extra needed</span>
            )}
          </div>
        </div>
      </div>

      {selectedShop && missingIngredients.length > 0 && (
  <div className="shopping-suggestion">
    <div className="shopping-icon">
      <Store size={18} />
    </div>

    <div>
      <strong>Shopping suggestion</strong>

      <p>
        Check {selectedShop.name} for {missingPreview}
        {missingIngredients.length > 4 ? "..." : ""}.
      </p>

      {primaryGrocerySuggestion && (
        <div className="shopping-detail">
          <span>
            Best store type: {primaryGrocerySuggestion.recommended_store_type}
          </span>

          <span>
            {primaryGrocerySuggestion.shopping_note}
          </span>
        </div>
      )}
    </div>

    <span className={getShopBadgeClass(selectedShop.type)}>
      {selectedShop.type}
    </span>
  </div>
)}

      <button className="instructions-button" onClick={onToggle}>
        {isOpen ? "Hide instructions" : "View instructions"}
        <ArrowRight size={16} className={isOpen ? "rotate" : ""} />
      </button>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            className="instructions-panel"
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
          >
            {instructions.length > 0 ? (
              instructions.map((step, stepIndex) => (
                <div className="instruction-step" key={`${stepIndex}-${step}`}>
                  <span>{stepIndex + 1}</span>
                  <p>{step}</p>
                </div>
              ))
            ) : (
              <p>No instructions available for this recipe.</p>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </motion.article>
  );
}

function getDifficultyLabel(difficulty) {
  const normalized = String(difficulty || "").toLowerCase();

  if (normalized === "easy") {
    return "Easy to complete";
  }

  if (normalized === "hard") {
    return "Needs specific ingredients";
  }

  return "Needs a few groceries";
}

function getDifficultyClass(difficulty) {
  const normalized = String(difficulty || "").toLowerCase();

  if (normalized === "easy") {
    return "difficulty-easy";
  }

  if (normalized === "hard") {
    return "difficulty-hard";
  }

  return "difficulty-medium";
}

function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export default App;