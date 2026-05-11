import ApartmentRoundedIcon from '@mui/icons-material/ApartmentRounded';
import Avatar from '@mui/material/Avatar';
import Box from '@mui/material/Box';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Chip from '@mui/material/Chip';
import Container from '@mui/material/Container';
import CssBaseline from '@mui/material/CssBaseline';
import Divider from '@mui/material/Divider';
import Stack from '@mui/material/Stack';
import Typography from '@mui/material/Typography';
import AppTheme from '../../shared-theme/AppTheme';
import AppAppBar from '../marketing-page/components/AppAppBar';
import Footer from '../marketing-page/components/Footer';

/**
 * NOTE: All copy below is placeholder. Replace the strings in `THESIS`,
 * `AUTHORS`, `ADVISER`, `EXAMINER`, and `ACKNOWLEDGEMENTS` with the real
 * thesis title, abstract, names, roles, affiliations, and supporting
 * institutions once finalised.
 */

type Person = {
  name: string;
  role: string;
  bio: string;
  initials: string;
};

const THESIS = {
  title: 'Chexit: Deep-Learning-Assisted Tuberculosis Screening from Chest X-Rays',
  year: '2026',
  institution: 'Your University · Department of Computer Science',
  abstract:
    'Chexit is an undergraduate thesis project exploring AI-assisted tuberculosis ' +
    'screening from chest radiographs. A lung-segmentation U-Net normalises the ' +
    'input field of view, a CNN ensemble (MobileNetV2, EfficientNetB2, DenseNet121) ' +
    'predicts a TB risk score, and Score-CAM overlays highlight the regions that ' +
    'drove each prediction. This page describes the project and the people behind it.',
  tags: [
    'Deep learning',
    'Chest X-ray',
    'Tuberculosis',
    'Explainable AI',
    'Score-CAM',
    'Undergraduate thesis',
  ],
};

const AUTHORS: Person[] = [
  {
    name: 'Jane Doe',
    role: 'Co-author · Frontend & API',
    bio: 'Designed and built the React + MUI dashboard and the FastAPI surface that fronts the inference pipeline.',
    initials: 'JD',
  },
  {
    name: 'John Smith',
    role: 'Co-author · Machine Learning',
    bio: 'Trained the segmentation + classification ensemble and authored the Score-CAM explainability layer.',
    initials: 'JS',
  },
];

const ADVISER: Person = {
  name: 'Dr. Alex Reyes',
  role: 'Faculty Adviser · Department of Computer Science',
  bio: 'Provided guidance on study design, evaluation methodology, and clinical relevance throughout the project.',
  initials: 'AR',
};

const EXAMINER: Person = {
  name: 'Dr. Sam Cruz',
  role: 'Thesis Examiner · Department of Computer Science',
  bio: 'Reviewed the thesis defense and provided critical feedback on methodology, results, and the final manuscript.',
  initials: 'SC',
};

const ACKNOWLEDGEMENTS = {
  intro:
    'With the help of the following institutions, mentors, and collaborators who ' +
    'provided datasets, computing resources, clinical guidance, and feedback ' +
    'throughout the project — thank you.',
  institutions: [
    { name: 'Institution One', initials: 'I1' },
    { name: 'Institution Two', initials: 'I2' },
    { name: 'Institution Three', initials: 'I3' },
    { name: 'Institution Four', initials: 'I4' },
    { name: 'Institution Five', initials: 'I5' },
  ],
};

function LogoPlaceholder({ name, initials }: { name: string; initials: string }) {
  return (
    <Box
      role="img"
      aria-label={name}
      sx={(theme) => ({
        position: 'relative',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        textAlign: 'center',
        gap: 1,
        height: 96,
        px: 1.5,
        borderRadius: 2,
        border: '1px dashed',
        borderColor: 'divider',
        backgroundColor: 'rgba(15,23,42,0.02)',
        color: 'text.secondary',
        transition: 'border-color 120ms ease, background-color 120ms ease, color 120ms ease',
        '&:hover': {
          borderStyle: 'solid',
          borderColor: 'text.primary',
          color: 'text.primary',
          backgroundColor: 'rgba(15,23,42,0.04)',
        },
        ...theme.applyStyles('dark', {
          backgroundColor: 'rgba(255,255,255,0.02)',
          '&:hover': {
            backgroundColor: 'rgba(255,255,255,0.04)',
          },
        }),
      })}
    >
      <Stack direction="row" spacing={1} alignItems="center">
        <ApartmentRoundedIcon fontSize="small" />
        <Typography
          variant="caption"
          sx={{ fontWeight: 700, letterSpacing: 1.5, fontSize: 12 }}
        >
          {initials}
        </Typography>
      </Stack>
      <Typography variant="caption" sx={{ fontSize: 11, lineHeight: 1.2 }}>
        {name}
      </Typography>
    </Box>
  );
}

function PersonCard({
  person,
  accentColor = 'primary.main',
}: {
  person: Person;
  accentColor?: string;
}) {
  return (
    <Card
      variant="outlined"
      sx={{
        borderRadius: 3,
        height: '100%',
        transition: 'border-color 120ms ease, box-shadow 120ms ease',
        '&:hover': {
          borderColor: 'text.primary',
        },
      }}
    >
      <CardContent sx={{ p: { xs: 3, sm: 3.5 } }}>
        <Stack direction="row" spacing={2} alignItems="center" sx={{ mb: 1.5 }}>
          <Avatar
            sx={{
              bgcolor: accentColor,
              width: 56,
              height: 56,
              fontWeight: 700,
              fontSize: 18,
            }}
          >
            {person.initials}
          </Avatar>
          <Box sx={{ minWidth: 0 }}>
            <Typography variant="h6" sx={{ fontWeight: 700, lineHeight: 1.2 }}>
              {person.name}
            </Typography>
            <Typography variant="body2" sx={{ color: 'text.secondary', mt: 0.25 }}>
              {person.role}
            </Typography>
          </Box>
        </Stack>
        <Typography variant="body2" sx={{ color: 'text.secondary', lineHeight: 1.65 }}>
          {person.bio}
        </Typography>
      </CardContent>
    </Card>
  );
}

export default function AboutPage(props: { disableCustomTheme?: boolean }) {
  return (
    <AppTheme {...props}>
      <CssBaseline enableColorScheme />
      <AppAppBar />
      <Box
        sx={(theme) => ({
          width: '100%',
          backgroundRepeat: 'no-repeat',
          backgroundImage:
            'radial-gradient(ellipse 80% 50% at 50% -20%, hsl(210, 100%, 90%), transparent)',
          ...theme.applyStyles('dark', {
            backgroundImage:
              'radial-gradient(ellipse 80% 50% at 50% -20%, hsl(210, 100%, 16%), transparent)',
          }),
        })}
      >
        <Container
          maxWidth="lg"
          sx={{ pt: { xs: 12, sm: 16 }, pb: { xs: 4, sm: 6 } }}
        >
          <Stack
            spacing={2}
            useFlexGap
            sx={{ alignItems: 'center', textAlign: 'center', maxWidth: 760, mx: 'auto' }}
          >
            <Typography
              variant="overline"
              sx={{ color: 'primary.main', letterSpacing: 2, fontWeight: 600 }}
            >
              ABOUT THE PROJECT
            </Typography>
            <Typography
              variant="h1"
              sx={{
                fontSize: 'clamp(2.4rem, 7vw, 3rem)',
                lineHeight: 1.1,
              }}
            >
              The Chexit&nbsp;
              <Typography
                component="span"
                variant="h1"
                sx={(theme) => ({
                  fontSize: 'inherit',
                  color: 'primary.main',
                  ...theme.applyStyles('dark', { color: 'primary.light' }),
                })}
              >
                thesis
              </Typography>
            </Typography>
            <Typography
              sx={{
                textAlign: 'center',
                color: 'text.secondary',
                width: { sm: '100%', md: '90%' },
              }}
            >
              An AI-assisted tuberculosis screening tool built as an undergraduate
              thesis project. This page describes the work and the people behind it.
            </Typography>
          </Stack>
        </Container>
      </Box>

      <Container maxWidth="lg" sx={{ pb: { xs: 6, md: 8 } }}>
        {/* Thesis card */}
        <Card variant="outlined" sx={{ borderRadius: 3, mb: { xs: 4, md: 6 } }}>
          <CardContent sx={{ p: { xs: 3, sm: 4 } }}>
            <Typography
              variant="overline"
              sx={{ color: 'text.secondary', letterSpacing: 1.5 }}
            >
              THESIS
            </Typography>
            <Typography
              variant="h5"
              sx={{ fontWeight: 700, mt: 0.75, mb: 1.5, lineHeight: 1.25 }}
            >
              {THESIS.title}
            </Typography>
            <Stack
              direction="row"
              spacing={1}
              useFlexGap
              flexWrap="wrap"
              sx={{ mb: 2.5, rowGap: 1 }}
            >
              <Chip label={THESIS.year} size="small" />
              <Chip label={THESIS.institution} size="small" />
            </Stack>
            <Typography variant="body1" sx={{ color: 'text.secondary', lineHeight: 1.75 }}>
              {THESIS.abstract}
            </Typography>
            <Stack
              direction="row"
              spacing={0.75}
              sx={{ mt: 2.5, flexWrap: 'wrap', gap: 0.75 }}
            >
              {THESIS.tags.map((tag) => (
                <Chip key={tag} label={tag} size="small" variant="outlined" />
              ))}
            </Stack>
          </CardContent>
        </Card>

        {/* Team section */}
        <Box sx={{ mb: 2 }}>
          <Typography
            variant="overline"
            sx={{ color: 'text.secondary', letterSpacing: 1.5 }}
          >
            TEAM
          </Typography>
          <Typography variant="h4" sx={{ fontWeight: 700, mt: 0.5 }}>
            Authors, adviser &amp; examiner
          </Typography>
          <Typography variant="body2" sx={{ color: 'text.secondary', mt: 1 }}>
            The thesis was researched and built by two students under the guidance of one faculty adviser, with the defense reviewed by an examiner.
          </Typography>
        </Box>

        <Box
          sx={{
            display: 'grid',
            gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr' },
            gap: { xs: 2, sm: 2.5 },
            mt: 3,
          }}
        >
          {AUTHORS.map((author) => (
            <PersonCard key={author.name} person={author} />
          ))}
        </Box>

        <Divider sx={{ my: { xs: 4, sm: 5 } }}>
          <Chip label="Adviser" size="small" sx={{ px: 1 }} />
        </Divider>

        <Box sx={{ display: 'flex', justifyContent: 'center' }}>
          <Box sx={{ width: '100%', maxWidth: 540 }}>
            <PersonCard person={ADVISER} accentColor="secondary.main" />
          </Box>
        </Box>

        <Divider sx={{ my: { xs: 4, sm: 5 } }}>
          <Chip label="Examiner" size="small" sx={{ px: 1 }} />
        </Divider>

        <Box sx={{ display: 'flex', justifyContent: 'center' }}>
          <Box sx={{ width: '100%', maxWidth: 540 }}>
            <PersonCard person={EXAMINER} accentColor="info.main" />
          </Box>
        </Box>

        {/* Acknowledgements */}
        <Box sx={{ mt: { xs: 6, sm: 8 } }}>
          <Typography
            variant="overline"
            sx={{ color: 'text.secondary', letterSpacing: 1.5 }}
          >
            ACKNOWLEDGEMENTS
          </Typography>
          <Typography variant="h4" sx={{ fontWeight: 700, mt: 0.5 }}>
            With the help of
          </Typography>
          <Typography
            variant="body1"
            sx={{ color: 'text.secondary', mt: 1.25, maxWidth: 760, lineHeight: 1.7 }}
          >
            {ACKNOWLEDGEMENTS.intro}
          </Typography>

          <Box
            sx={{
              mt: { xs: 3, sm: 4 },
              display: 'grid',
              gridTemplateColumns: {
                xs: 'repeat(2, 1fr)',
                sm: 'repeat(3, 1fr)',
                md: 'repeat(5, 1fr)',
              },
              gap: { xs: 1.5, sm: 2 },
            }}
          >
            {ACKNOWLEDGEMENTS.institutions.map((logo) => (
              <LogoPlaceholder
                key={logo.name}
                name={logo.name}
                initials={logo.initials}
              />
            ))}
          </Box>

          <Typography
            variant="caption"
            sx={{ display: 'block', color: 'text.secondary', mt: 2, fontStyle: 'italic' }}
          >
            Placeholder logos — replace with the real institutional marks.
          </Typography>
        </Box>
      </Container>

      <Footer />
    </AppTheme>
  );
}
